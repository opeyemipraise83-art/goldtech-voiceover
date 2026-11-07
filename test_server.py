"""Local test of the GOLDTECH Voice engine API with a stubbed model.
Validates reference prep (ffmpeg decode + trim + normalize), chunking, the job
lifecycle, and every endpoint the web app calls — without needing the GPU model.
"""
import os, sys, time, shutil
import numpy as np
import soundfile as sf

# make static ffmpeg discoverable
try:
    import static_ffmpeg; static_ffmpeg.add_paths()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
os.environ["GT_VOICE_BASE"] = "/tmp/gt_voice_test"
shutil.rmtree("/tmp/gt_voice_test", ignore_errors=True)

import server

SR = 24000
server.SR = SR
server.DEVICE = "cpu"

class StubModel:
    """Mimics ChatterboxTurboTTS.prepare_conditionals + .generate."""
    sr = SR
    def prepare_conditionals(self, path, exaggeration=0.0, norm_loudness=True):
        y, sr = sf.read(path)
        assert sr == SR and len(y) / sr > 5.0, "reference must be >5s at 24kHz"
        self.conditioned = True
    def generate(self, text, temperature=0.8, repetition_penalty=1.2, top_p=0.95, norm_loudness=True, **kwargs):
        import torch
        # ~0.4s of audio per word, a voiced-ish tone so normalization has signal
        n = max(int(SR * 0.12 * len(text.split())), int(SR * 0.3))
        t = np.arange(n) / SR
        y = (0.3 * np.sin(2 * np.pi * 180 * t) * np.exp(-t * 0.5)).astype(np.float32)
        return torch.from_numpy(y).unsqueeze(0)

server.MODEL = StubModel()

# ---- unit: chunking ----
assert server.chunk_text("", 220) == []
short = "Hello there. This is a test."
assert server.chunk_text(short, 220) == [short]
long = ("This is a fairly long sentence that keeps going and going. " * 12).strip()
chunks = server.chunk_text(long, 100)
assert all(len(c) <= 100 for c in chunks), [len(c) for c in chunks]
assert " ".join(chunks).split() == long.split(), "chunking must preserve words"
one_long_word_sentence = "word " * 80
chunks2 = server.chunk_text(one_long_word_sentence, 60)
assert all(len(c) <= 60 for c in chunks2)
print("chunking: OK", f"({len(chunks)} chunks for long text)")

# ---- build a fake 16s reference WAV (speech-like tone bursts + silence edges) ----
t = np.arange(int(SR * 16)) / SR
voice = 0.4 * np.sin(2 * np.pi * 160 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
ref = np.concatenate([np.zeros(SR), voice, np.zeros(SR)]).astype(np.float32)  # 18s w/ silence
sf.write("/tmp/ref_test.wav", ref, SR, subtype="PCM_16")

from fastapi.testclient import TestClient
client = TestClient(server.app)

# health
r = client.get("/api/health"); assert r.status_code == 200
h = r.json(); assert h["ok"] and h["warm"] and h["model"] == "chatterbox-turbo"
print("health:", h)

# consent required
r = client.post("/api/jobs", data={"script": "Hi", "consent": "false"},
                files={"reference": ("ref.wav", open("/tmp/ref_test.wav","rb"), "audio/wav")})
assert r.status_code == 400, r.text
print("consent gate: OK")

# too-short reference rejected
sf.write("/tmp/ref_short.wav", np.zeros(int(SR*2), dtype=np.float32), SR)
r = client.post("/api/jobs", data={"script": "Hello world, this is my cloned voice speaking.", "consent": "true"},
                files={"reference": ("short.wav", open("/tmp/ref_short.wav","rb"), "audio/wav")})
assert r.status_code == 200
jid = r.json()["job_id"]
for _ in range(300):
    s = client.get(f"/api/jobs/{jid}").json()
    if s["status"] in ("done","error","cancelled"): break
    time.sleep(0.2)
assert s["status"] == "error" and "6s" in s["message"], s
print("short-reference rejection: OK ->", s["message"][:60])

# happy path
script = "Welcome to GOLDTECH Voice. This is my own voice, cloned accurately. " * 6
r = client.post("/api/jobs", data={"script": script, "consent": "true",
                                   "temperature": "0.8", "pause": "0.3", "max_chars": "120", "seed": "42"},
                files={"reference": ("ref.wav", open("/tmp/ref_test.wav","rb"), "audio/wav")})
assert r.status_code == 200, r.text
jid = r.json()["job_id"]
print("job created:", jid)
for _ in range(400):
    s = client.get(f"/api/jobs/{jid}").json()
    if s["status"] in ("done","error","cancelled"): break
    time.sleep(0.2)
assert s["status"] == "done", s
assert s["chunks_total"] >= 2 and s["chunks_done"] == s["chunks_total"]
assert s["reference"]["seconds"] >= 6
print("job done:", {k: s[k] for k in ("status","duration","chunks_total","reference")})

# audio + mp3 endpoints
r = client.get(f"/api/jobs/{jid}/audio"); assert r.status_code == 200
assert r.headers["content-type"].startswith("audio/wav")
wav_bytes = r.content
y, sr = sf.read(__import__("io").BytesIO(wav_bytes))
assert sr == SR and len(y) > SR
peak = float(np.max(np.abs(y))); assert peak <= 0.991, peak
print(f"audio endpoint: OK ({len(wav_bytes)} bytes, {len(y)/sr:.1f}s, peak {peak:.3f})")
r = client.get(f"/api/jobs/{jid}/mp3")
print("mp3 endpoint:", r.status_code, r.headers.get("content-type"))
assert r.status_code == 200 and r.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")

# 409 while busy: start a job then immediately try another
r1 = client.post("/api/jobs", data={"script": "Another take.", "consent":"true"},
                 files={"reference": ("ref.wav", open("/tmp/ref_test.wav","rb"), "audio/wav")})
assert r1.status_code == 200
r2 = client.post("/api/jobs", data={"script": "Should conflict.", "consent":"true"},
                 files={"reference": ("ref.wav", open("/tmp/ref_test.wav","rb"), "audio/wav")})
assert r2.status_code == 409, r2.status_code
print("concurrency 409: OK")
# cancel it
jid2 = r1.json()["job_id"]
client.post(f"/api/jobs/{jid2}/cancel")
for _ in range(300):
    s2 = client.get(f"/api/jobs/{jid2}").json()
    if s2["status"] in ("done","error","cancelled"): break
    time.sleep(0.2)
print("cancel: OK ->", s2["status"])

print("\nALL ENGINE TESTS PASSED ✅")
