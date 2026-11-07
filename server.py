# GOLDTECH Voice — Colab engine server
# Runs inside Google Colab. Loads Chatterbox Turbo once, keeps it warm on the GPU,
# and exposes a small JSON API that the GOLDTECH Voice web app talks to through a
# Cloudflare tunnel. One job at a time; progress is polled.
#
# This file is written to /content/gt_voice/server.py by the notebook, then imported.

import os
import re
import sys
import time
import uuid
import math
import shutil
import threading
import subprocess
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("GT_VOICE_BASE", "/content/gt_voice"))
JOBS = BASE / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

# Filled in by the notebook's run cell after the model loads.
MODEL = None
DEVICE = "cpu"
SR = 24000
MODEL_NAME = "chatterbox-turbo"

MAX_SCRIPT_CHARS = 20000
REF_MIN_SEC = 6.0      # model asserts > 5s; we require a little more for accuracy
REF_MAX_SEC = 30.0     # longer references don't help and slow conditioning
TARGET_LUFS = -16.0    # loudness of the finished voiceover

_jobs = {}
_jobs_lock = threading.Lock()
_gpu_lock = threading.Lock()   # serialize GPU work


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Fallback for Kaggle / Studio environments where apt-get isn't available.
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_wav(src: Path, dst: Path, sr: int = 24000):
    """Decode any uploaded audio (webm/opus/mp3/m4a/wav…) to mono PCM WAV."""
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg is not installed in this runtime")
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not dst.exists():
        raise RuntimeError(f"Could not decode audio: {p.stderr.strip()[:300]}")


def prepare_reference(src: Path, dst: Path) -> dict:
    """Convert, trim silence, normalize, and length-check the reference clip."""
    import librosa
    import soundfile as sf

    tmp = dst.with_suffix(".raw.wav")
    convert_to_wav(src, tmp, sr=SR)
    y, _ = librosa.load(str(tmp), sr=SR, mono=True)
    try:
        tmp.unlink()
    except OSError:
        pass

    if y.size == 0:
        raise ValueError("Reference audio is empty")

    # Trim leading/trailing silence so the model hears speech, not room tone.
    yt, _ = librosa.effects.trim(y, top_db=30, frame_length=1024, hop_length=256)
    if yt.size < SR * 1.0:
        yt = y  # trimming was too aggressive; keep the original

    dur = len(yt) / SR
    if dur < REF_MIN_SEC:
        raise ValueError(
            f"Reference is only {dur:.1f}s after trimming silence. "
            f"Record at least {REF_MIN_SEC:.0f}s of clear speech (15–20s is ideal)."
        )
    if dur > REF_MAX_SEC:
        yt = yt[: int(REF_MAX_SEC * SR)]
        dur = REF_MAX_SEC

    # Loudness-normalize to the model's expected -27 LUFS ourselves, keeping the
    # array strictly float32. Chatterbox's internal norm_loudness can promote the
    # waveform to float64 (pyloudnorm returns a numpy.float64 gain), which causes
    # "expected scalar type Float but found Double" on CUDA. We disable that
    # internal step below and hand the model a clean float32 reference instead.
    normalized = False
    try:
        import pyloudnorm as ln
        meter = ln.Meter(SR)
        loud = meter.integrated_loudness(yt)
        if math.isfinite(loud):
            gain = np.float32(10.0 ** ((-27.0 - float(loud)) / 20.0))
            yt = yt * gain
            normalized = True
    except Exception:
        pass

    peak = float(np.max(np.abs(yt))) or 1.0
    if not normalized:
        yt = yt * np.float32(0.891 / peak)   # fallback: peak-normalize to -1 dBFS
    elif peak > 0.99:
        yt = yt * np.float32(0.99 / peak)    # safety limiter only

    yt = np.ascontiguousarray(yt, dtype=np.float32)
    sf.write(str(dst), yt, SR, subtype="PCM_16")
    return {"seconds": round(dur, 2), "sample_rate": SR}


def chunk_text(text: str, max_chars: int):
    """Split a script into <= max_chars chunks on sentence/phrase boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""

    def push(piece):
        nonlocal cur
        piece = piece.strip()
        if not piece:
            return
        if cur and len(cur) + 1 + len(piece) <= max_chars:
            cur = (cur + " " + piece).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = piece

    for s in sentences:
        if len(s) <= max_chars:
            push(s)
            continue
        # Long sentence: break on commas/semicolons/colons, then hard-wrap words.
        for part in re.split(r"(?<=[,;:])\s+", s):
            if len(part) <= max_chars:
                push(part)
                continue
            words, buf = part.split(), ""
            for w in words:
                if buf and len(buf) + 1 + len(w) > max_chars:
                    push(buf)
                    buf = w
                else:
                    buf = (buf + " " + w).strip()
            if buf:
                push(buf)
    if cur:
        chunks.append(cur)
    return chunks


def _fade(y: np.ndarray, ms: float = 6.0) -> np.ndarray:
    n = min(int(SR * ms / 1000.0), len(y) // 2)
    if n <= 0:
        return y
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = y.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def _normalize_loudness(y: np.ndarray) -> np.ndarray:
    try:
        import pyloudnorm as ln
        meter = ln.Meter(SR)
        loud = meter.integrated_loudness(y)
        if math.isfinite(loud):
            gain = 10.0 ** ((TARGET_LUFS - loud) / 20.0)
            y = y * gain
    except Exception:
        pass
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def _float32_conditionals(conds):
    """Force any floating conditioning tensors to float32 (keep int tokens intact)."""
    import torch
    if conds is None:
        return
    t3 = getattr(conds, "t3", None)
    if t3 is not None:
        for k, v in list(vars(t3).items()):
            if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32:
                setattr(t3, k, v.to(torch.float32))
    gen = getattr(conds, "gen", None)
    if isinstance(gen, dict):
        for k, v in list(gen.items()):
            if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32:
                gen[k] = v.to(torch.float32)


# --------------------------------------------------------------------------- #
# Job runner
# --------------------------------------------------------------------------- #
def _run_job(job_id: str):
    import soundfile as sf
    import torch

    with _jobs_lock:
        job = _jobs[job_id]
    root = JOBS / job_id
    ref_path = root / "reference.wav"
    out_wav = root / "voiceover.wav"
    out_mp3 = root / "voiceover.mp3"

    try:
        with _jobs_lock:
            job["status"] = "running"
            job["message"] = "Preparing reference voice…"

        ref_info = prepare_reference(root / job["ref_upload_name"], ref_path)
        with _jobs_lock:
            job["reference"] = ref_info

        chunks = chunk_text(job["script"], job["max_chars"])
        if not chunks:
            raise ValueError("Script is empty")
        with _jobs_lock:
            job["chunks_total"] = len(chunks)
            job["message"] = f"Cloning voice… 0/{len(chunks)}"

        if job["seed"] is not None:
            torch.manual_seed(int(job["seed"]))

        segments = []
        pause_samples = int(SR * job["pause"])

        # Serialize GPU access; keep the model warm between jobs.
        with _gpu_lock:
            if MODEL is None:
                raise RuntimeError("Model is not loaded yet")
            # Condition once on the reference, then reuse for every chunk so the
            # voice stays identical across the whole script. norm_loudness=False:
            # we already loudness-normalized the reference to -27 LUFS in float32.
            MODEL.prepare_conditionals(str(ref_path), exaggeration=0.0, norm_loudness=False)
            _float32_conditionals(getattr(MODEL, "conds", None))

            for i, text in enumerate(chunks):
                if job.get("cancel"):
                    raise RuntimeError("Cancelled")
                with _jobs_lock:
                    job["message"] = f"Cloning voice… {i}/{len(chunks)}"
                wav = MODEL.generate(
                    text,
                    temperature=job["temperature"],
                    repetition_penalty=1.2,
                    top_p=0.95,
                    norm_loudness=False,
                )
                _float32_conditionals(getattr(MODEL, "conds", None))
                y = wav.detach().to(torch.float32).squeeze(0).cpu().numpy()
                if y.size < SR * 0.2:
                    continue  # skip a degenerate chunk
                segments.append(_fade(y))
                if i < len(chunks) - 1 and pause_samples > 0:
                    segments.append(np.zeros(pause_samples, dtype=np.float32))
                with _jobs_lock:
                    job["chunks_done"] = i + 1
                    job["progress"] = round((i + 1) / len(chunks), 4)
                    job["message"] = f"Cloning voice… {i + 1}/{len(chunks)}"

        if not segments:
            raise RuntimeError("Generation produced no audio")

        with _jobs_lock:
            job["message"] = "Mixing and normalizing…"
        audio = _normalize_loudness(np.concatenate(segments))
        sf.write(str(out_wav), audio, SR, subtype="PCM_16")

        if _ffmpeg():
            p = subprocess.run(
                [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(out_wav), "-codec:a", "libmp3lame", "-qscale:a", "2", str(out_mp3)],
                capture_output=True, text=True,
            )
            if p.returncode != 0:
                out_mp3 = None
        else:
            out_mp3 = None

        with _jobs_lock:
            job["status"] = "done"
            job["progress"] = 1.0
            job["message"] = "Voiceover ready"
            job["duration"] = round(len(audio) / SR, 2)
            job["audio"] = str(out_wav)
            job["mp3"] = str(out_mp3) if out_mp3 and out_mp3.exists() else None
            job["finished"] = time.time()

    except Exception as e:
        with _jobs_lock:
            cancelled = job.get("cancel")
            job["status"] = "cancelled" if cancelled else "error"
            job["message"] = "Cancelled" if cancelled else str(e)
            job["error"] = None if cancelled else str(e)
            job["finished"] = time.time()


# --------------------------------------------------------------------------- #
# Web API
# --------------------------------------------------------------------------- #
def create_app():
    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="GOLDTECH Voice Engine")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # personal ephemeral tunnel; no credentials
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return ("<html><body style='font-family:sans;padding:2rem'>"
                "<h2>GOLDTECH Voice engine is running.</h2>"
                "<p>Open the GOLDTECH Voice web app and paste this tunnel URL.</p>"
                "</body></html>")

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "model": MODEL_NAME,
            "device": DEVICE,
            "sample_rate": SR,
            "warm": MODEL is not None,
            "gpu": DEVICE == "cuda",
        }

    @app.post("/api/jobs")
    async def create_job(
        script: str = Form(...),
        reference: UploadFile = File(...),
        temperature: float = Form(0.8),
        pause: float = Form(0.35),
        max_chars: int = Form(220),
        seed: str = Form(""),
        consent: str = Form("false"),
    ):
        if consent != "true":
            raise HTTPException(400, "You must confirm you own the voice or have permission to clone it.")
        if MODEL is None:
            raise HTTPException(503, "Model is still loading. Wait a moment and try again.")
        script = script.strip()
        if not script:
            raise HTTPException(400, "Script is empty.")
        if len(script) > MAX_SCRIPT_CHARS:
            raise HTTPException(400, f"Script too long (max {MAX_SCRIPT_CHARS} characters).")
        if not (0.1 <= temperature <= 1.5):
            raise HTTPException(400, "Temperature must be 0.1–1.5.")
        if not (0.0 <= pause <= 2.0):
            raise HTTPException(400, "Pause must be 0–2 seconds.")
        if not (60 <= max_chars <= 500):
            raise HTTPException(400, "Chunk size must be 60–500.")
        seed_val = None
        if seed.strip():
            try:
                seed_val = int(seed.strip())
            except ValueError:
                raise HTTPException(400, "Seed must be a whole number.")

        with _jobs_lock:
            running = [j for j in _jobs.values() if j["status"] in ("queued", "running")]
            if running:
                raise HTTPException(409, "A job is already running. Wait for it to finish or cancel it.")

        job_id = uuid.uuid4().hex[:12]
        root = JOBS / job_id
        root.mkdir(parents=True, exist_ok=True)
        upload_name = "reference" + Path(reference.filename or "ref.wav").suffix.lower()
        if upload_name == "reference":
            upload_name = "reference.webm"
        dest = root / upload_name
        data = await reference.read()
        if len(data) > 40 * 1024 * 1024:
            raise HTTPException(400, "Reference audio too large (max 40 MB).")
        dest.write_bytes(data)

        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "script": script,
            "ref_upload_name": upload_name,
            "temperature": float(temperature),
            "pause": float(pause),
            "max_chars": int(max_chars),
            "seed": seed_val,
            "chunks_total": 0,
            "chunks_done": 0,
            "created": time.time(),
            "cancel": False,
        }
        with _jobs_lock:
            _jobs[job_id] = job
        threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(404, "Job not found")
            return {k: v for k, v in job.items()
                    if k not in ("script", "ref_upload_name", "cancel")}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(404, "Job not found")
            job["cancel"] = True
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/audio")
    def job_audio(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job or job["status"] != "done" or not job.get("audio"):
            raise HTTPException(404, "Audio not ready")
        return FileResponse(job["audio"], media_type="audio/wav", filename="goldtech-voiceover.wav")

    @app.get("/api/jobs/{job_id}/mp3")
    def job_mp3(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job or job["status"] != "done" or not job.get("mp3"):
            raise HTTPException(404, "MP3 not ready")
        return FileResponse(job["mp3"], media_type="audio/mpeg", filename="goldtech-voiceover.mp3")

    return app


app = create_app()
