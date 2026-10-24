"""Recapping Toolz backend.
video -> Whisper transcript -> Gemini Burmese translation -> edge-tts Burmese voice
-> mix with video -> burned-in Burmese subtitles.
"""
import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------- config (env vars) ----------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "whisper-large-v3-turbo")
MAX_MB = int(os.environ.get("MAX_MB", "150"))
MAX_MIN = int(os.environ.get("MAX_MIN", "10"))
SHORT_SIDE = int(os.environ.get("SHORT_SIDE", "720"))  # 0 = keep original size
WORKERS = int(os.environ.get("WORKERS", "1"))
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "20"))
KEEP_HOURS = 6

VOICES = {"my-MM-ThihaNeural", "my-MM-NilarNeural"}  # male, female
EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SR = 24000
MAX_TEMPO = 1.5

BASE = Path(__file__).parent
WORK = BASE / "work"
FONTS = BASE / "fonts"
WORK.mkdir(exist_ok=True)

app = FastAPI()
pool = ThreadPoolExecutor(max_workers=WORKERS)
JOBS: dict = {}


# ---------- helpers ----------
def run(cmd, cwd=None) -> bytes:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="ignore")[-400:])
    return r.stdout


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type", "-of", "json", str(path)])
    d = json.loads(out)
    dur = float(d.get("format", {}).get("duration") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
    return dur, has_audio


def upd(job, **kw):
    job.update(kw)


# ---------- 1. transcript ----------
def transcribe(job, audio, total):
    """Groq-hosted Whisper (free tier)."""
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    r = None
    for attempt in range(3):
        with open(audio, "rb") as fh:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                files={"file": ("a.mp3", fh, "audio/mpeg")},
                data={"model": GROQ_MODEL, "response_format": "verbose_json",
                      "timestamp_granularities[]": "segment", "temperature": "0"},
                timeout=300,
            )
        if r.status_code == 200:
            break
        time.sleep(5 * (attempt + 1))
    if r is None or r.status_code != 200:
        raise RuntimeError("Transcription failed: " + (r.text[:120] if r is not None else ""))
    out = []
    for seg in r.json().get("segments", []):
        t = seg.get("text", "").strip()
        if t and seg.get("no_speech_prob", 0) < 0.7:
            out.append({"start": float(seg["start"]), "end": float(seg["end"]), "src": t})
    if not out:
        raise ValueError("No speech found")
    upd(job, p=35)
    return out


# ---------- 2. Burmese translation (Gemini free tier) ----------
PROMPT = (
    "Translate each line to natural, spoken Burmese (Myanmar) for a video recap "
    "narration. Keep the meaning, be concise, use everyday words, and put a space "
    "between phrases. Return ONLY a JSON array of exactly {n} strings in the same "
    "order.\n\n{data}"
)


def gemini(prompt: str) -> str:
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
        },
        timeout=120,
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def translate(job, segs):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    step = 25
    for i in range(0, len(segs), step):
        chunk = [s["src"] for s in segs[i:i + step]]
        res = None
        for attempt in range(3):
            try:
                data = json.dumps(chunk, ensure_ascii=False)
                res = json.loads(gemini(PROMPT.format(n=len(chunk), data=data)))
                if isinstance(res, list) and len(res) == len(chunk):
                    break
                res = None
            except Exception:
                res = None
            time.sleep(2 * (attempt + 1))
        if res is None:
            raise RuntimeError("Translation failed")
        for s, t in zip(segs[i:i + step], res):
            s["my"] = str(t).strip()
        upd(job, p=35 + 15 * min((i + step) / len(segs), 1))


# ---------- 3. Burmese voice (edge-tts) ----------
async def _tts_all(job, segs, voice):
    sem = asyncio.Semaphore(4)
    done = 0

    async def one(s):
        nonlocal done
        async with sem:
            for a in range(3):
                try:
                    await edge_tts.Communicate(s["my"], voice).save(str(s["mp3"]))
                    break
                except Exception:
                    if a == 2:
                        raise
                    await asyncio.sleep(1.5)
        done += 1
        upd(job, p=50 + 30 * done / len(segs))

    await asyncio.gather(*[one(s) for s in segs])


def decode(path, tempo=1.0):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if tempo > 1.01:
        cmd += ["-filter:a", f"atempo={tempo:.3f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"]
    return np.frombuffer(run(cmd), dtype=np.float32)


def build_voice(segs, dur, out_wav):
    """Fit each clip to its slot (speed up to MAX_TEMPO), never overlap, save wav."""
    clips, cursor = [], 0.0
    for i, s in enumerate(segs):
        a = decode(s["mp3"])
        length = len(a) / SR
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else s["end"] + 3
        slot = max(max(s["end"], nxt) - s["start"], 0.3)
        tempo = min(max(length / slot, 1.0), MAX_TEMPO)
        if tempo > 1.01:
            a = decode(s["mp3"], tempo)
            length = len(a) / SR
        start = max(s["start"], cursor)
        s["a_start"], s["a_end"] = start, start + length
        cursor = start + length + 0.05
        clips.append((start, a))
    buf = np.zeros(int((max(dur, cursor) + 1) * SR), dtype=np.float32)
    for start, a in clips:
        p = int(start * SR)
        buf[p:p + len(a)] = a
    buf = buf[: int(dur * SR)]
    peak = float(np.abs(buf).max()) or 1.0
    if peak > 0.95:
        buf *= 0.95 / peak
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------- 4. subtitles + final render ----------
def ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segs, path):
    lines = []
    for i, s in enumerate(segs, 1):
        lines += [str(i), f"{ts(s['a_start'])} --> {ts(s['a_end'])}", s["my"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


STYLE = ("FontName=Noto Sans Myanmar,FontSize=16,Bold=1,PrimaryColour=&H00FFFFFF,"
         "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=30")


def render(d: Path, src: Path, has_audio: bool, dur: float):
    sub = f"subtitles=subs.srt:fontsdir={FONTS}:force_style='{STYLE}'"
    if SHORT_SIDE:  # shrink big videos so a small free server can render them
        sub = (f"scale='if(gt(iw,ih),-2,min({SHORT_SIDE},iw))':"
               f"'if(gt(iw,ih),min({SHORT_SIDE},ih),-2)'," + sub)
    voice = "[1:a]pan=stereo|c0=c0|c1=c0"
    if has_audio:  # original sound ducked under the Burmese voice
        fc = (f"[0:v]{sub}[v];[0:a]volume=0.12[o];{voice}[t];"
              "[t][o]amix=inputs=2:duration=first:dropout_transition=0,"
              "volume=2,alimiter=limit=0.95[a]")
    else:
        fc = f"[0:v]{sub}[v];{voice}[a]"
    run(["ffmpeg", "-y", "-i", str(src), "-i", "voice.wav", "-filter_complex", fc,
         "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "out.mp4"], cwd=d)


# ---------- pipeline ----------
def process(jid):
    job = JOBS[jid]
    d = WORK / jid
    try:
        upd(job, status="running", stage="listen", p=2)
        src = next(d.glob("src.*"))
        dur, has_audio = probe(src)
        if not has_audio:
            raise ValueError("No audio in this video")
        if dur > MAX_MIN * 60:
            raise ValueError(f"Video is longer than {MAX_MIN} minutes")
        run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "48k", str(d / "a.mp3")])
        segs = transcribe(job, d / "a.mp3", dur)

        upd(job, stage="translate", p=35)
        translate(job, segs)
        segs = [s for s in segs if s.get("my")]
        if not segs:
            raise ValueError("Nothing to speak")
        for i, s in enumerate(segs):
            s["mp3"] = d / f"s{i}.mp3"

        upd(job, stage="voice", p=50)
        asyncio.run(_tts_all(job, segs, job["voice"]))

        upd(job, stage="render", p=82)
        build_voice(segs, dur, d / "voice.wav")
        write_srt(segs, d / "subs.srt")
        render(d, src, has_audio, dur)
        for f in d.glob("s*.mp3"):
            f.unlink()
        upd(job, status="done", stage="done", p=100)
    except Exception as e:  # noqa: BLE001
        upd(job, status="error", error=str(e)[:200])


# ---------- API ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), voice: str = Form("my-MM-ThihaNeural")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in EXTS:
        raise HTTPException(400, "Unsupported file type")
    if voice not in VOICES:
        voice = "my-MM-ThihaNeural"
    if sum(j["status"] in ("queued", "running") for j in JOBS.values()) >= QUEUE_MAX:
        raise HTTPException(429, "Server is busy. Try again in a few minutes.")
    jid = uuid.uuid4().hex[:12]
    d = WORK / jid
    d.mkdir()
    size = 0
    with open(d / f"src{ext}", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"File is larger than {MAX_MB} MB")
            f.write(chunk)
    JOBS[jid] = {"status": "queued", "stage": "queued", "p": 0, "error": "",
                 "voice": voice, "t": time.time()}
    pool.submit(process, jid)
    return {"id": jid}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return {k: j[k] for k in ("status", "stage", "p", "error")}


@app.get("/api/file/{jid}/{name}")
def get_file(jid: str, name: str):
    if name not in ("out.mp4", "subs.srt") or jid not in JOBS:
        raise HTTPException(404)
    p = WORK / jid / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4" if name.endswith(".mp4") else "text/plain")


def cleaner():
    while True:
        time.sleep(1800)
        cutoff = time.time() - KEEP_HOURS * 3600
        for jid, j in list(JOBS.items()):
            if j["t"] < cutoff and j["status"] not in ("queued", "running"):
                shutil.rmtree(WORK / jid, ignore_errors=True)
                JOBS.pop(jid, None)


threading.Thread(target=cleaner, daemon=True).start()
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
