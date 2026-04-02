"""
ClipFind Backend v2 — Smart trimming, per-user indexes, context-enriched search
"""
import os, json, uuid, shutil, subprocess, hashlib, time, threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "AIzaSyDy2YvX51IpOZb6GSDQUqnwsjvT3P4ncVg")
STRIPE_LINK  = "https://buy.stripe.com/6oU5kE3Ay1H6d5I71BgYU01"

BASE    = Path(__file__).parent
USERS   = BASE / "users.json"
UPLOADS = BASE / "uploads"
CLIPS   = BASE / "clips"
INDEXES = BASE / "indexes"

PLANS = {
    "free":  {"credits": 1,   "name": "Free",  "price": 0},
    "pro":   {"credits": 50,  "name": "Pro",   "price": 4900},
    "team":  {"credits": 200, "name": "Team",  "price": 14900},
}

BLOCKED_QUERIES = {
    "porn","pornography","nude","nudity","sex","nsfw","explicit","adult",
    "genitals","penis","vagina","breast","undress","fetish","hentai","erotic"
}

def load_users():
    if USERS.exists(): return json.loads(USERS.read_text())
    return {}

def save_users(u):
    USERS.write_text(json.dumps(u, indent=2))

def get_user(token):
    return load_users().get(token)

def create_user(email, plan="free"):
    users = load_users()
    # Check existing
    for t, u in users.items():
        if u["email"] == email:
            return u
    token = hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()[:32]
    users[token] = {
        "email": email, "plan": plan, "token": token,
        "credits_used": 0, "credits_total": PLANS[plan]["credits"],
        "videos": [], "created_at": datetime.now().isoformat()
    }
    save_users(users)
    return users[token]

def get_base_env():
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = GEMINI_KEY
    env["HOME"] = str(Path.home())
    env["PATH"] = f"{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin"
    return env

# ── Precise trim via Gemini Vision ──
def precise_trim(video_path: Path, query: str, start_sec: int, end_sec: int) -> tuple:
    """Ask Gemini Vision to find exact start+end within a chunk."""
    try:
        from google import genai
        from google.genai import types

        chunk_sec = end_sec - start_sec
        
        # Extract chunk as proper re-encoded MP4 (not stream copy — ensures valid container)
        chunk_path = Path(f"/tmp/chunk_{uuid.uuid4().hex[:8]}.mp4")
        r = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-ss", str(start_sec), "-t", str(chunk_sec),
            "-c:v", "libx264", "-c:a", "aac",
            "-vf", "scale=640:-2",  # downscale for faster upload
            "-crf", "28", str(chunk_path), "-y"
        ], capture_output=True, timeout=30)

        if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
            chunk_path.unlink(missing_ok=True)
            return start_sec, end_sec

        video_bytes = chunk_path.read_bytes()
        chunk_path.unlink(missing_ok=True)

        client = genai.Client(api_key=GEMINI_KEY)
        
        prompt = f"""This is a {chunk_sec}-second video clip.
Find the precise moment matching this description: "{query}"

Rules:
- Return ONLY valid JSON, nothing else
- start_offset and end_offset are seconds FROM THE START of this clip (0 to {chunk_sec})
- Keep the clip tight — just the key moment, min 2s, max 15s
- If the described event is not visible, set confidence below 0.4

{{"start_offset": <number>, "end_offset": <number>, "confidence": <0.0-1.0>}}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                prompt
            ]
        )
        
        import re as _re
        text = response.text.strip()
        m = _re.search(r'\{[^{}]+\}', text)
        if m:
            d = json.loads(m.group())
            conf = float(d.get("confidence", 0))
            if conf >= 0.4:
                os_  = max(0, float(d.get("start_offset", 0)))
                oe_  = min(chunk_sec, float(d.get("end_offset", chunk_sec)))
                # 1s buffer each side
                ps = start_sec + max(0, os_ - 1)
                pe = start_sec + min(chunk_sec, oe_ + 1)
                if pe - ps >= 2:
                    return int(ps), int(pe)
    except Exception as e:
        pass  # fallback to full chunk
    return start_sec, end_sec

# ── Index in background ──
def index_video_bg(vid_path: Path, token: str, vid_id: str):
    user_index = INDEXES / token
    user_index.mkdir(parents=True, exist_ok=True)

    # Set per-user index via env var
    env = get_base_env()
    env["SENTRYSEARCH_DB"] = str(user_index)

    # Create temp dir with just this video
    tmp_dir = BASE / "tmp_index" / vid_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    link = tmp_dir / vid_path.name
    if not link.exists():
        shutil.copy2(str(vid_path), str(link))

    try:
        result = subprocess.run(
            ["sentrysearch", "index", str(tmp_dir)],
            env=env, capture_output=True, text=True, timeout=600
        )
        status = "ready" if result.returncode == 0 else "error"
        error  = result.stderr[:300] if result.returncode != 0 else ""
    except Exception as e:
        status, error = "error", str(e)
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    users = load_users()
    for v in users.get(token, {}).get("videos", []):
        if v["id"] == vid_id:
            v["status"] = status
            v["indexed_at"] = datetime.now().isoformat()
            if error: v["indexing_error"] = error
    save_users(users)
    
    # Send email notification on success
    if status == "ready":
        try:
            email = users.get(token, {}).get("email", "")
            if email:
                vid_name = next((v["filename"] for v in users[token]["videos"] if v["id"] == vid_id), "your video")
                subprocess.run([
                    "gog", "mail", "send",
                    "--account", "clawkaan@gmail.com",
                    "--to", email,
                    "--subject", f"✅ {vid_name} is ready to search — ClipFind",
                    "--body", f"Your video \"{vid_name}\" has been indexed and is ready to search.\n\nOpen ClipFind: http://localhost:8103/app\n\nHappy searching,\nClipFind 🎬"
                ], capture_output=True, timeout=15)
        except Exception:
            pass

app = FastAPI(title="ClipFind v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0", "gemini": bool(GEMINI_KEY)}


ADMIN_CREDS = {
    "kt": {"password": "kt", "token": "demo1234567890abcdef1234567890ab"}
}

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = ADMIN_CREDS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(401, "Invalid username or password")
    token = user["token"]
    account = get_user(token)
    if not account:
        raise HTTPException(404, "Account not found")
    return {"token": token, "plan": account["plan"], "email": account["email"]}

@app.post("/api/signup")
async def signup(email: str = Form(...)):
    user = create_user(email)
    return {"token": user["token"], "plan": user["plan"],
            "credits": user["credits_total"], "message": "Welcome to ClipFind!"}

@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    token: str = Form(...),
    description: str = Form(default="")
):
    user = get_user(token)
    if not user: raise HTTPException(401, "Invalid token")
    if user["credits_used"] >= user["credits_total"]:
        raise HTTPException(402, "Credit limit reached. Upgrade at clipfind.ai")

    # Dedup by hash
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    for v in user["videos"]:
        if v.get("hash") == file_hash:
            return {"video_id": v["id"], "status": v["status"], "message": "Already indexed (duplicate)"}

    vid_id = str(uuid.uuid4())[:8]
    user_dir = UPLOADS / token
    user_dir.mkdir(parents=True, exist_ok=True)
    vid_path = user_dir / f"{vid_id}_{file.filename}"
    vid_path.write_bytes(content)

    users = load_users()
    users[token]["videos"].append({
        "id": vid_id, "filename": file.filename, "hash": file_hash,
        "description": description, "path": str(vid_path),
        "status": "indexing", "uploaded_at": datetime.now().isoformat(),
    })
    users[token]["credits_used"] = round(users[token]["credits_used"] + 0.1, 2)
    save_users(users)

    background_tasks.add_task(index_video_bg, vid_path, token, vid_id)
    return {"video_id": vid_id, "status": "indexing", "message": "Indexing in background (~8 min/hr of footage)"}

@app.post("/api/search")
async def search(
    token: str = Form(...),
    video_id: str = Form(...),
    query: str = Form(...),
    max_results: int = Form(default=4)
):
    # Content filter
    import re as _re
    q_lower = query.lower()
    for blocked in BLOCKED_QUERIES:
        # Word-boundary match to avoid false positives like "naked truth"
        if _re.search(r'\b' + _re.escape(blocked) + r'\b', q_lower):
            raise HTTPException(400, "Query not allowed. ClipFind is for professional footage search.")

    user = get_user(token)
    if not user: raise HTTPException(401, "Invalid token")

    video = next((v for v in user["videos"] if v["id"] == video_id), None)
    if not video: raise HTTPException(404, "Video not found")
    if video["status"] != "ready":
        raise HTTPException(400, f"Video is {video['status']} — try again shortly")

    # Enrich query with video context
    desc = video.get("description", "")
    enriched = f"In: {desc}. Find: {query}" if desc else query

    clip_dir = CLIPS / token / video_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    env = get_base_env()
    # Try per-user index first, fall back to global
    user_index = INDEXES / token
    global_index = Path.home() / ".sentrysearch" / "db"
    global_db_root = Path.home() / ".sentrysearch"
    
    if user_index.exists() and any(user_index.iterdir()):
        env["SENTRYSEARCH_DB"] = str(user_index)
    elif global_index.exists():
        env["SENTRYSEARCH_DB"] = str(global_db_root)
    else:
        env["SENTRYSEARCH_DB"] = str(user_index)

    max_r = max(1, min(4, max_results))

    try:
        result = subprocess.run(
            ["sentrysearch", "search", enriched,
             "--output-dir", str(clip_dir),
             "--save-top", str(max_r),
             "--threshold", "0.45"],
            env=env, capture_output=True, text=True, timeout=60, input="y\n"
        )
        output = result.stdout + result.stderr

        # Parse results
        import re
        matches = re.findall(r'#(\d+)\s+\[([0-9.]+)\]\s+(.+?\.mp4)\s+@\s+([\d:]+)-([\d:]+)', output)

        results = []
        vid_path = Path(video["path"])

        for m in matches:
            rank, score_s, fname, ts_start, ts_end = m
            score = float(score_s)

            # Convert timestamp to seconds
            def ts_to_sec(ts):
                parts = ts.split(":")
                return sum(int(p) * 60**(len(parts)-i-1) for i, p in enumerate(parts))

            s = ts_to_sec(ts_start)
            e = ts_to_sec(ts_end)

            # Precise trim via Gemini Vision
            ps, pe = precise_trim(vid_path, query, s, e)

            # Export precise clip
            clip_name = f"clip_{rank}_{ps}s-{pe}s.mp4"
            clip_out = clip_dir / clip_name
            subprocess.run([
                "ffmpeg", "-i", str(vid_path),
                "-ss", str(ps), "-to", str(pe),
                "-c:v", "libx264", "-c:a", "aac", "-crf", "28",
                str(clip_out), "-y"
            ], capture_output=True, timeout=30)

            results.append({
                "rank": int(rank),
                "score": score,
                "start_sec": ps,
                "end_sec": pe,
                "duration_sec": pe - ps,
                "timestamp": f"{ts_start} → precise: {ps}s–{pe}s",
                "clip_url": f"/api/clip/{token}/{video_id}/{clip_name}" if clip_out.exists() else None,
            })

        return {
            "query": query,
            "enriched_query": enriched if enriched != query else None,
            "results": results,
            "total": len(results),
            "message": f"Found {len(results)} match(es)" if results else "No confident matches found"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/clip/{token}/{video_id}/{filename}")
async def get_clip(token: str, video_id: str, filename: str):
    user = get_user(token)
    if not user: raise HTTPException(401)
    clip_path = CLIPS / token / video_id / filename
    if not clip_path.exists(): raise HTTPException(404)
    return FileResponse(clip_path, media_type="video/mp4",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.get("/api/videos/{token}")
async def list_videos(token: str):
    user = get_user(token)
    if not user: raise HTTPException(401)
    return {
        "videos": user["videos"],
        "plan": user["plan"],
        "credits_used": user["credits_used"],
        "credits_total": user["credits_total"]
    }


@app.get("/api/status/{token}/{video_id}")
async def video_status(token: str, video_id: str):
    user = get_user(token)
    if not user: raise HTTPException(401)
    video = next((v for v in user["videos"] if v["id"] == video_id), None)
    if not video: raise HTTPException(404)
    return {
        "video_id": video_id,
        "status": video["status"],
        "filename": video["filename"],
        "indexed_at": video.get("indexed_at"),
        "error": video.get("indexing_error"),
    }


@app.delete("/api/videos/{token}/{video_id}")
async def delete_video(token: str, video_id: str):
    user = get_user(token)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    videos = user.get("videos", [])
    video = next((v for v in videos if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Remove from user videos list
    user["videos"] = [v for v in videos if v["id"] != video_id]
    save_user(token, user)
    
    # Clean up files
    import shutil
    for path in [
        UPLOAD_DIR / f"{video_id}*",
        INDEX_DIR / video_id,
    ]:
        import glob
        for f in glob.glob(str(path)):
            try:
                if os.path.isdir(f): shutil.rmtree(f)
                else: os.remove(f)
            except: pass
    
    return {"status": "deleted", "video_id": video_id}

@app.delete("/api/videos/{token}")
async def delete_all_videos(token: str):
    user = get_user(token)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    import shutil
    for video in user.get("videos", []):
        vid_id = video["id"]
        import glob
        for path in [str(UPLOAD_DIR / f"{vid_id}*"), str(INDEX_DIR / vid_id)]:
            for f in glob.glob(path):
                try:
                    if os.path.isdir(f): shutil.rmtree(f)
                    else: os.remove(f)
                except: pass
    
    user["videos"] = []
    save_user(token, user)
    return {"status": "all_deleted"}

@app.get("/", response_class=HTMLResponse)
async def landing():
    from fastapi.responses import Response
    return Response(
        content=(BASE / "static" / "index.html").read_text(),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )

@app.get("/app", response_class=HTMLResponse)
async def dashboard():
    from fastapi.responses import Response
    return Response(
        content=(BASE / "static" / "app.html").read_text(),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )

@app.get("/{fname}")
async def static_file(fname: str):
    p = BASE / "static" / fname
    if p.exists(): return FileResponse(p)
    raise HTTPException(404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8105)
