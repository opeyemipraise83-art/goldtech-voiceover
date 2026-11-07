#!/usr/bin/env python3
"""Assemble the self-contained GOLDTECH Voice notebook.

Works on Google Colab, Kaggle, and generic GPU hosts (Lightning AI Studio,
SageMaker Studio Lab, or a local machine). Platform is auto-detected at runtime.

Kaggle-safe design:
- Never downgrades the host's preinstalled numpy/scipy/sklearn/numba/torch stack.
- Runs GPU checks, import smoke tests, and the engine itself in FRESH Python
  subprocesses, so a stale numpy already loaded in the notebook kernel cannot
  break the model import.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SERVER = (HERE / "server.py").read_text()

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

# Shared platform detector injected into the cells that need a base directory.
DETECT = '''import os
def _gt_base():
    if os.path.isdir("/kaggle/working"):      # Kaggle
        return "/kaggle/working/gt_voice"
    if os.path.isdir("/content"):             # Google Colab
        return "/content/gt_voice"
    return os.path.abspath("gt_voice")        # Lightning / SageMaker / local
BASE = _gt_base()
os.makedirs(BASE, exist_ok=True)
os.environ["GT_VOICE_BASE"] = BASE
def _gt_platform():
    if os.path.isdir("/kaggle/working"): return "Kaggle"
    if os.path.isdir("/content"): return "Google Colab"
    return "GPU host"
'''

INTRO = """# 🎙️ GOLDTECH Voice — free GPU voice cloning

Clone **your own voice** and read any script on a free GPU.
Powered by **Chatterbox Turbo** (MIT-licensed, by Resemble AI).

**Works on:** Google Colab · Kaggle · Lightning AI Studio · SageMaker Studio Lab · any CUDA machine.
The notebook auto-detects the platform.

## How to run (once per session)
1. Turn on the GPU:
   - **Colab:** Runtime → Change runtime type → **T4 GPU**.
   - **Kaggle:** Settings → Accelerator → **GPU T4 x2** (or P100), and turn **Internet ON**.
   - **Lightning / SageMaker:** pick a GPU instance/runtime.
2. **If you ran an older version of this notebook, restart the session first**
   (Kaggle: Run → Restart Session and Clear Output; Colab: Runtime → Restart session).
3. **Run all** cells top to bottom.
4. (Colab only) Approve the Google Drive prompt to cache the model — optional.
5. Copy the **`https://….trycloudflare.com`** URL printed by the last cell.
6. Open the **GOLDTECH Voice** web app → **Connect engine** → paste the URL.

> The URL changes each session. Keep the notebook tab open while generating.
> If a session dies, re-run the last cell for a fresh URL.

**Only clone a voice you own or have explicit permission to use.**
"""

# Fresh-process probes. These are injected as repr() literals so the notebook
# kernel itself never imports torch/numpy during setup.
GPU_CODE = (
    "import torch, json; "
    "print(json.dumps({'torch': torch.__version__, "
    "'cuda': torch.cuda.is_available(), "
    "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))"
)
SMOKE_CODE = """import numpy, scipy.sparse, sklearn, torch, torchaudio, transformers, librosa
import soundfile, pyloudnorm, omegaconf, einops, onnxruntime, fastapi, uvicorn, tqdm
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model, GenerationMixin
from chatterbox.tts_turbo import ChatterboxTurboTTS
print('SMOKE OK | numpy', numpy.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)
"""

GPU_SETUP_TEMPLATE = '''# @title 1) Check GPU + install engine (Kaggle-safe)
import sys, subprocess, shutil, os, sysconfig, urllib.request, importlib.metadata
from pathlib import Path
''' + DETECT + '''
print("Platform:", _gt_platform(), "| work dir:", BASE)

# Kaggle's system Python can be PEP 668 "externally managed": pip refuses to
# install unless we pass --break-system-packages. Detect the marker and add it.
_marker = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
_PIP_FLAGS = ["--break-system-packages"] if os.path.exists(_marker) else []

def pip(*args):
    base = [sys.executable, "-m", "pip", "install", "--no-input"] + _PIP_FLAGS
    r = subprocess.run(base + list(args), capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 and "externally-managed" in out.lower() and "--break-system-packages" not in base:
        r = subprocess.run(base + ["--break-system-packages"] + list(args), capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print(out[-4000:])   # show the REAL pip error, not a bare exit code
        raise RuntimeError(f"pip install failed ({r.returncode}): {' '.join(args)}")
    tail = [l for l in out.strip().splitlines() if l.strip()][-2:]
    print("\\n".join(tail))

# Quick internet check — most Kaggle pip failures are simply Internet = OFF.
try:
    urllib.request.urlopen("https://pypi.org", timeout=10)
    print("Internet: OK (PyPI reachable)")
except Exception:
    print("⚠️ Cannot reach PyPI. Turn ON Internet in the Kaggle right-sidebar Settings, then re-run.")

# Check the GPU in a FRESH interpreter. We deliberately do NOT import torch or
# numpy into the notebook kernel here: changing packages after they are loaded
# is what caused the earlier numpy.ufunc / LlamaModel import crash.
_gpu_code = __GPU_CODE__
r = subprocess.run([sys.executable, "-c", _gpu_code], capture_output=True, text=True)
if r.returncode == 0:
    import json as _json
    info = _json.loads(r.stdout.strip().splitlines()[-1])
    print("PyTorch:", info["torch"], "| CUDA available:", info["cuda"])
    if info["cuda"]:
        print("GPU:", info["gpu"])
    else:
        print("⚠️ No GPU detected. Enable a GPU accelerator in notebook settings. Continuing on CPU (very slow).")
else:
    print("⚠️ Could not check torch:", ((r.stderr or "") + (r.stdout or ""))[-800:])

# Chatterbox pins torch==2.6.0 which would replace the host's CUDA build.
# Install the prebuilt PyPI wheel with --no-deps (no git/build step), keeping the
# preinstalled GPU torch. Then add only the real runtime dependencies.
print("\\nInstalling Chatterbox (prebuilt wheel, no-deps, keeps your GPU torch)…")
pip("--no-deps", "chatterbox-tts==0.1.7")

# Lock the host's preinstalled scientific/CUDA stack so pip cannot downgrade or
# replace it. This is the key fix for the numpy.ufunc '__module__' crash: we no
# longer force numpy==2.2.6 over Kaggle's newer numpy.
constraints = []
for pkg in ["numpy", "scipy", "scikit-learn", "numba", "llvmlite", "torch", "torchaudio"]:
    try:
        constraints.append(f"{pkg}=={importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        pass
cfile = Path(BASE) / "constraints.txt"
cfile.write_text("\\n".join(constraints) + "\\n")
print("Locked host packages:", ", ".join(constraints) or "(none found)")

print("Installing runtime dependencies…")
pip("-c", str(cfile),
    "transformers==5.2.0", "diffusers==0.29.0", "librosa==0.11.0", "s3tokenizer",
    "conformer==0.3.2", "resemble-perth", "pyloudnorm", "omegaconf", "einops",
    "onnxruntime", "soundfile", "fastapi", "uvicorn", "python-multipart",
    "imageio-ffmpeg", "tqdm")

# Verify every import in a FRESH interpreter before we ever load the model.
_smoke = __SMOKE_CODE__
r = subprocess.run([sys.executable, "-c", _smoke], capture_output=True, text=True)
if r.returncode != 0:
    print(((r.stdout or "") + (r.stderr or ""))[-4000:])
    raise RuntimeError("Import smoke test failed — see the real error above.")
print(r.stdout.strip())

# ffmpeg: prefer system binary; otherwise the imageio-ffmpeg wheel provides one.
if not shutil.which("ffmpeg"):
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    except Exception:
        print("apt-get unavailable — using imageio-ffmpeg bundled binary.")

# cloudflared gives us a free public HTTPS URL for the engine.
CF = os.path.join(BASE, "cloudflared")
if not os.path.exists(CF):
    print("Downloading cloudflared…")
    subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    "-O", CF], check=True)
    os.chmod(CF, 0o755)
print("\\n✅ Setup complete.")
'''
GPU_SETUP = GPU_SETUP_TEMPLATE.replace("__GPU_CODE__", repr(GPU_CODE)).replace("__SMOKE_CODE__", repr(SMOKE_CODE))

CACHE = '''# @title 2) (Optional) Cache the model for faster restarts
import os
''' + DETECT + '''
plat = _gt_platform()
if plat == "Google Colab":
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        cache = "/content/drive/MyDrive/GOLDTECH-Voice/hf_cache"
        os.makedirs(cache, exist_ok=True)
        os.environ["HF_HOME"] = cache
        print("✅ Model cache →", cache)
    except Exception as e:
        print("Drive not mounted (", e, "). Model re-downloads each session.")
elif plat == "Kaggle":
    cache = os.path.join(BASE, "hf_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    print("✅ Model cache →", cache, "(persists in this session's working dir)")
else:
    cache = os.path.join(BASE, "hf_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    print("✅ Model cache →", cache)
'''

# Write the server with a RELATIVE path so it lands in the right place on every
# platform (Colab cwd=/content, Kaggle cwd=/kaggle/working, else cwd).
WRITE_SERVER = "%%writefile gt_voice/server.py\n" + SERVER
SERVER_MD = ("## 3) Write the engine server\n"
             "Run this cell — it writes `server.py` (the FastAPI engine) into the working dir. "
             "Expected output: `Writing gt_voice/server.py`.")

# The engine runs in its own fresh Python process (see RUN cell). This avoids
# any stale numpy/torch already imported by the notebook kernel.
ENGINE_MAIN = '''"""GOLDTECH Voice engine entrypoint — runs in a fresh Python process."""
import os, sys, time, re, threading, subprocess, urllib.request
from pathlib import Path

BASE = os.environ.get("GT_VOICE_BASE") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ.setdefault("HF_HOME", os.path.join(BASE, "hf_cache"))

import torch
import numpy as np
import uvicorn
from chatterbox.tts_turbo import ChatterboxTurboTTS
import server

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox Turbo on {device.upper()}… (first run downloads ~1.5 GB)", flush=True)
model = ChatterboxTurboTTS.from_pretrained(device=device)
print(f"✅ Model ready · sample rate {model.sr} Hz", flush=True)

# Force a strict float32 pipeline. Kaggle's torch/pyloudnorm stack can otherwise
# promote audio to float64 and trigger "expected scalar type Float but found Double".
torch.set_default_dtype(torch.float32)
for _m in (model.t3, model.s3gen, model.ve):
    _m.float()
_orig_norm = model.norm_loudness
def _norm_loudness_f32(wav, sr, target_lufs=-27):
    out = _orig_norm(wav, sr, target_lufs)
    try:
        if isinstance(out, np.ndarray):
            return out.astype(np.float32)
        if torch.is_tensor(out):
            return out.to(torch.float32)
    except Exception:
        pass
    return out
model.norm_loudness = _norm_loudness_f32
print("✅ float32 audio pipeline locked", flush=True)

server.MODEL = model
server.DEVICE = device
server.SR = model.sr

PORT = 8000
config = uvicorn.Config(server.app, host="0.0.0.0", port=PORT, log_level="warning")
srv = uvicorn.Server(config)
threading.Thread(target=srv.run, daemon=True).start()

for _ in range(120):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
print("✅ Engine API is up on port", PORT, flush=True)

CF = os.path.join(BASE, "cloudflared")
tunnel = subprocess.Popen(
    [CF, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
public_url = None
start = time.time()
while time.time() - start < 120:
    line = tunnel.stdout.readline()
    if not line:
        time.sleep(0.2)
        continue
    m = re.search(r"https://[A-Za-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        public_url = m.group(0)
        break

if public_url:
    Path(BASE, "url.txt").write_text(public_url)
    print("\\n" + "=" * 70, flush=True)
    print("✅ GOLDTECH VOICE ENGINE READY", flush=True)
    print("\\n   👉  ", public_url, "\\n", flush=True)
    print("Paste this URL into the web app under 'Connect engine'.", flush=True)
    print("=" * 70, flush=True)
else:
    print("⚠️ Tunnel URL not detected yet. Check the cloudflared output / engine.log.", flush=True)
    print("   On Kaggle, make sure Internet is ON in Settings.", flush=True)

# Keep the process alive so the API + tunnel stay up.
while True:
    time.sleep(3600)
'''
WRITE_ENGINE = "%%writefile gt_voice/engine_main.py\n" + ENGINE_MAIN
ENGINE_MD = ("## 4) Write the engine launcher\n"
             "Run this cell — it writes `engine_main.py`, which loads the model and opens the tunnel "
             "in a fresh Python process. Expected output: `Writing gt_voice/engine_main.py`.")

RUN = '''# @title 5) Start engine + open tunnel  ← copy the URL it prints
import os, sys, time, subprocess
from pathlib import Path
''' + DETECT + '''
engine_py = Path(BASE) / "engine_main.py"
log_path = Path(BASE) / "engine.log"
url_path = Path(BASE) / "url.txt"
if not engine_py.exists():
    raise RuntimeError("engine_main.py not found — run the previous cell first.")

# Stop a previous engine from an earlier run of this cell, if any.
prev = globals().get("GT_ENGINE_PROC")
if prev is not None and prev.poll() is None:
    prev.terminate()
    try:
        prev.wait(timeout=5)
    except Exception:
        prev.kill()

if url_path.exists():
    url_path.unlink()

env = os.environ.copy()
env["GT_VOICE_BASE"] = BASE
env.setdefault("HF_HOME", str(Path(BASE) / "hf_cache"))

log_f = open(log_path, "w")
proc = subprocess.Popen([sys.executable, str(engine_py)],
                        stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=BASE)
GT_ENGINE_PROC = proc
print("Starting engine in a fresh Python process…")
print("Log:", log_path)

def _lines():
    try:
        return log_path.read_text(errors="replace").splitlines()
    except Exception:
        return []

seen = 0
start = time.time()
public_url = None
while time.time() - start < 900:   # up to 15 min for the first model download
    if proc.poll() is not None:
        print("\\n".join(_lines()[-60:]))
        raise RuntimeError(f"Engine exited early (code {proc.returncode}). See the log above.")
    lines = _lines()
    if len(lines) > seen:
        print("\\n".join(lines[seen:]))
        seen = len(lines)
    if url_path.exists():
        public_url = url_path.read_text().strip()
        break
    time.sleep(2)

print("\\n" + "=" * 70)
if public_url:
    print("✅ GOLDTECH VOICE ENGINE READY")
    print("\\n   👉  ", public_url, "\\n")
    print("Paste this URL into the web app under 'Connect engine'.")
else:
    print("⚠️ Timed out waiting for the tunnel URL. Recent log:")
    print("\\n".join(_lines()[-40:]))
    print("If the model is still downloading, wait and re-run this cell.")
print("=" * 70)
print("Keep this notebook tab open while generating. To stop, restart the session.")
'''

OUTRO = """## Tips for an accurate clone
- **Reference clip:** 15–20 seconds of you speaking naturally, quiet room, no music/echo. Record straight into the web app.
- **Expressiveness:** start at **0.8**. Lower (0.5–0.7) = steadier, more faithful; higher (1.0+) = more dramatic.
- **Chunk size:** ~220 characters keeps the voice consistent on long scripts.
- The engine conditions on your reference **once** and reuses it for every chunk.
- Output is loudness-normalized to −16 LUFS, saved as WAV + MP3.

## Platform notes
- **Kaggle:** ~30 GPU hrs/week. Turn **Internet ON** (Settings) or the tunnel/model download fails. If cloudflared is blocked, use Colab or Lightning instead.
- **Lightning AI Studio:** free monthly credits; persistent storage. Run these cells in a notebook or adapt to a script.
- **SageMaker Studio Lab:** ~4 GPU hrs/day, persistent storage, no credit card.
- **Colab:** free tier disconnects/throttles; re-run the last cell for a new URL.

## Troubleshooting
- **numpy / LlamaModel import error** → this version locks Kaggle's preinstalled numpy and runs the engine in a fresh process. Restart the session, then Run All.
- **"Model is still loading"** → wait for the last cell to finish; first download is ~1.5 GB.
- **409 / a job is already running** → wait or cancel in the app.
- **Tunnel not reachable** → session slept; re-run the last cell, re-paste URL.
- **Out of memory** → restart session, Run All (T4 16 GB is plenty for this 350M model).
- **Engine log:** `gt_voice/engine.log` in the working dir has the full model/tunnel output.
"""

nb = {
    "cells": [
        md(INTRO),
        code(GPU_SETUP),
        code(CACHE),
        md(SERVER_MD),
        code(WRITE_SERVER),
        md(ENGINE_MD),
        code(WRITE_ENGINE),
        code(RUN),
        md(OUTRO),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = HERE / "GOLDTECH_Voice.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out, f"({out.stat().st_size} bytes)")
