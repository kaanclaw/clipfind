"""
ClipFind Backend v2 — Smart trimming, per-user indexes, context-enriched search
"""
import os, json, uuid, shutil, subprocess, hashlib, time, threading
from pathlib import Path
import google.generativeai as genai
from google.generativeai import types
from datetime import datetime
from typing import Optional
import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "AIzaSyDy2YvX51IpOZb6GSDQUqnwsjvT3P4ncVg")
STRIPE_LINK  = "https://buy.stripe.com/6oU5kE3Ay1H6d5I71BgYU01"

import os as _os
BASE    = Path(__file__).parent
DATA    = Path(_os.environ.get("DATA_DIR", str(BASE)))
USERS   = DATA / "users.json"

# Ensure data directories exist and users.json is seeded on first run
for _d in [DATA / "uploads", DATA / "clips", DATA / "indexes"]:
    _d.mkdir(parents=True, exist_ok=True)
if not USERS.exists():
    import json as _json
    USERS.write_text(_json.dumps({
        "demo1234567890abcdef1234567890ab": {
            "email": "taskentbusiness@gmail.com",
            "plan": "pro", "credits_used": 0, "credits_total": 100, "videos": []
        }
    }))
UPLOADS = DATA / "uploads"
CLIPS   = DATA / "clips"
INDEXES = DATA / "indexes"

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
        ], capture_output=True, timeout=90)

        if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
            chunk_path.unlink(missing_ok=True)
            return start_sec, end_sec

        video_bytes = chunk_path.read_bytes()
        chunk_path.unlink(missing_ok=True)

        genai.configure(api_key=GEMINI_KEY)
        client = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""This is a {chunk_sec}-second video clip.
Find the precise moment matching this description: "{query}"

Rules:
- Return ONLY valid JSON, nothing else
- start_offset and end_offset are seconds FROM THE START of this clip (0 to {chunk_sec})
- Keep the clip tight — just the key moment, min 2s, max 15s
- If the described event is not visible, set confidence below 0.4

{{"start_offset": <number>, "end_offset": <number>, "confidence": <0.0-1.0>}}"""

        response = client.generate_content([
            {"mime_type": "video/mp4", "data": video_bytes},
            prompt
        ])
        
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
    """Index video by uploading to Gemini Files API and getting timestamped descriptions in one call."""
    import chromadb, time as _time, re as _re, json as _json
    
    user_index = INDEXES / token
    user_index.mkdir(parents=True, exist_ok=True)
    
    status = "error"
    error = ""
    gemini_file = None
    
    try:
        # Probe video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(vid_path)],
            capture_output=True, text=True, timeout=30
        )
        streams = _json.loads(probe.stdout).get("streams", [])
        duration = 0
        for s in streams:
            if s.get("codec_type") == "video":
                duration = float(s.get("duration", 0))
                break
        if not duration:
            duration = 600  # fallback 10min

        # Upload video to Gemini Files API
        genai.configure(api_key=GEMINI_KEY)
        gemini_file = genai.upload_file(str(vid_path), mime_type="video/mp4")
        
        # Wait for processing (max 3 min)
        for _ in range(36):
            if gemini_file.state.name == "ACTIVE":
                break
            _time.sleep(5)
            gemini_file = genai.get_file(gemini_file.name)
        
        if gemini_file.state.name != "ACTIVE":
            raise Exception(f"Gemini file processing failed: {gemini_file.state.name}")
        
        # Ask Gemini to describe every 5 seconds in one call
        interval = 5
        marks = list(range(0, int(duration), interval))
        marks_str = ", ".join(str(m) + "s" for m in marks[:120])  # max 120 marks
        
        # Try models in order of preference
        model_names = ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.0-flash-lite"]
        response = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                prompt = (
                    f"Analyze this dashcam video. For each of these timestamps: {marks_str}\n"
                    "Provide one line per timestamp in EXACTLY this format:\n"
                    "Ns: [description]\n"
                    "Where N is the timestamp in seconds. Describe: vehicle colors/types, "
                    "actions (turning, braking, crash, normal driving, etc), road/weather conditions. "
                    "Be specific about: red/blue/black/white cars, SUVs/sedans/trucks, "
                    "accidents/collisions/near-misses, snowy/wet/dry roads, intersections, "
                    "pedestrians, traffic lights. If nothing notable, say 'normal highway driving'."
                )
                response = model.generate_content([gemini_file, prompt])
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    _time.sleep(30)
                    continue
                raise
        
        if not response:
            raise Exception("All Gemini models quota exhausted")
        
        # Parse the timestamped descriptions
        client_db = chromadb.PersistentClient(path=str(user_index))
        collection_name = f"vid_{vid_id}"
        try:
            client_db.delete_collection(collection_name)
        except:
            pass
        collection = client_db.get_or_create_collection(
            collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        
        # Parse "Ns: description" format
        lines = response.text.strip().split("\n")
        docs, metas, ids = [], [], []
        
        for line in lines:
            m = _re.match(r'^(\d+)s:\s*(.+)', line.strip())
            if m:
                ts = int(m.group(1))
                desc = m.group(2).strip()
                if desc and desc != "N/A" and len(desc) > 5:
                    docs.append(desc)
                    metas.append({
                        "timestamp": ts,
                        "video_id": vid_id,
                        "duration": interval
                    })
                    ids.append(f"{vid_id}_{ts}")
        
        if docs:
            collection.add(documents=docs, metadatas=metas, ids=ids)
            status = "ready"
        else:
            status = "error"
            error = f"Could not parse descriptions. Response: {response.text[:200]}"
            
    except Exception as e:
        import traceback
        status, error = "error", traceback.format_exc()[-400:]
    finally:
        # Clean up Gemini file
        if gemini_file:
            try:
                genai.delete_file(gemini_file.name)
            except:
                pass

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

    user_index = INDEXES / token
    max_r = max(1, min(4, max_results))

    try:
        import chromadb
        client_db = chromadb.PersistentClient(path=str(user_index))
        collection_name = f"vid_{video_id}"
        try:
            collection = client_db.get_collection(collection_name)
        except Exception:
            raise HTTPException(400, "Video index not found — please re-upload and re-index this video")
        
        # Query the collection
        qr = collection.query(query_texts=[enriched], n_results=min(max_r * 6, 50))
        
        results = []
        vid_path = Path(video["path"])
        
        for i, (doc, meta, dist) in enumerate(zip(
            qr["documents"][0],
            qr["metadatas"][0], 
            qr["distances"][0]
        )):
            score = max(0, 1 - dist/2)  # cosine: dist 0=identical, 2=opposite
            if score < 0.65:
                continue
            
            # Hybrid filter: use synonym expansion to check description relevance
            import re as _re
            # Synonym map: query word → what to look for in description
            _synonyms = {
                'crash': ['crash','collision','accident','impact','struck','hitting','collid'],
                'accident': ['accident','collision','crash','impact','struck','incident'],
                'sedan': ['sedan','car','vehicle','automobile'],
                'snowy': ['snow','snowy','winter','icy','slippery','ice'],
                'snow': ['snow','snowy','winter','icy','slippery','ice'],
                'wet': ['wet','rain','rainy','slippery','puddle','water'],
                'rain': ['rain','rainy','wet','drizzle','storm'],
                'turning': ['turning','turn','turned','turning','veering','swerving','cutting'],
                'braking': ['braking','brake','braked','slowing','stopped','halt'],
                'highway': ['highway','freeway','interstate','motorway','expressway','multi'],
                'pedestrian': ['pedestrian','person','walker','walking','crosswalk','crossing'],
                'intersection': ['intersection','crossroads','junction','light','signal'],
                'merging': ['merging','merge','merged','changing','lane'],
                'collision': ['collision','crash','accident','impact','struck','hitting'],
            }
            
            # Generic words that appear in virtually all dashcam descriptions
            generic = {'driving','dashcam','vehicle','camera','footage','road','highway',
                       'lane','traffic','scene','car','truck','view','frame','image',
                       'captures','shows','find','this','that','from','with','into','onto',
                       'during','under','over','along','through','toward','ahead','behind',
                       'multi','speed','forward','clear','sunny','daytime','mph','the','and',
                       'for','are','was','had','has','its','can','may','not','all','also',
                       'via','one','two','three','four','five','day','time','open',
                       'traveling','moving','captured','recorded','shows','visible'}
            
            query_words_raw = set(_re.findall(r'\b[a-z]{3,}\b', query.lower()))
            specific_terms = query_words_raw - generic
            
            if specific_terms:
                desc_lower = doc.lower()
                # Expand each specific term with synonyms
                expanded = set()
                for term in specific_terms:
                    expanded.add(term)
                    expanded.update(_synonyms.get(term, []))
                
                # Must have at least ONE expanded term in description
                if not any(t in desc_lower for t in expanded):
                    continue
            
            ts = meta["timestamp"]
            chunk_dur = meta.get("duration", 10)
            s = max(0, ts - 2)
            e = ts + chunk_dur + 2
            
            # Use timestamp window directly (skip Gemini trim for speed)
            ps, pe = s, e
            
            # Export clip in background thread
            clip_name = f"clip_{i+1}_{ps}s-{pe}s.mp4"
            clip_out = clip_dir / clip_name
            
            if not clip_out.exists():
                import threading
                def make_clip(vp=vid_path, start=ps, dur=pe-ps, out=clip_out):
                    subprocess.run([
                        "ffmpeg", "-ss", str(start), "-i", str(vp),
                        "-t", str(dur), "-c:v", "libx264", "-c:a", "aac", "-crf", "28",
                        str(out), "-y"
                    ], capture_output=True, timeout=120)
                threading.Thread(target=make_clip, daemon=True).start()
            
            clip_url = f"/clips/{token}/{video_id}/{clip_name}"
            
            # Deduplicate: skip if within 15s of an existing result
            too_close = any(abs(ps - prev['start_sec']) < 15 for prev in results)
            if not too_close:
                results.append({
                    "rank": len(results) + 1,
                    "score": round(score, 3),
                    "start_sec": ps,
                    "end_sec": pe,
                    "duration_sec": pe - ps,
                    "description": doc[:100],
                    "clip_url": clip_url,
                })
            
            if len(results) >= max_r:
                break



    except Exception as e:
        import traceback
        raise HTTPException(500, f"Search failed: {traceback.format_exc()[-300:]}")

    return {"results": results}



@app.get("/api/videos/{token}")
async def list_videos(token: str):
    user = get_user(token)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid token")
    return {
        "plan": user.get("plan", "free"),
        "credits_used": user.get("credits_used", 0),
        "credits_total": user.get("credits_total", 10),
        "email": user.get("email", ""),
        "videos": [
            {
                "id": v["id"],
                "filename": v["filename"],
                "description": v.get("description", ""),
                "status": v["status"],
                "uploaded_at": v.get("uploaded_at", ""),
                "indexed_at": v.get("indexed_at"),
                "path": v.get("path", ""),
            }
            for v in user.get("videos", [])
        ]
    }

@app.get("/api/status/{token}/{video_id}")
async def video_status(token: str, video_id: str):
    user = get_user(token)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid token")
    video = next((v for v in user.get("videos", []) if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "video_id": video_id,
        "filename": video["filename"],
        "status": video["status"],
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
    user["videos"] = [v for v in videos if v["id"] != video_id]
    users = load_users()
    users[token] = user
    save_users(users)
    import shutil, glob
    for path in [str(UPLOADS / f"{video_id}*"), str(INDEXES / video_id)]:
        for f in glob.glob(path):
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
    import shutil, glob
    for video in user.get("videos", []):
        vid_id = video["id"]
        for path in [str(UPLOADS / f"{vid_id}*"), str(INDEXES / vid_id)]:
            for f in glob.glob(path):
                try:
                    if os.path.isdir(f): shutil.rmtree(f)
                    else: os.remove(f)
                except: pass
    user["videos"] = []
    users = load_users()
    users[token] = user
    save_users(users)
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

@app.get("/clips/{token}/{video_id}/{filename}")
async def serve_clip(token: str, video_id: str, filename: str):
    from fastapi.responses import FileResponse
    clip_path = CLIPS / token / video_id / filename
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(clip_path), media_type="video/mp4")
