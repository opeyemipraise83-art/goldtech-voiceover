# 🎙️ GOLDTECH Voice

Clone **your own voice** and read any script — running on a **free GPU** (Colab, Kaggle,
Lightning AI, or SageMaker Studio Lab), the same trick the TryAIToday site uses. No paid API, no server to rent.

- **Engine:** [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) (MIT license, by Resemble AI) — zero-shot voice cloning from a short sample.
- **Web app:** `index.html` — a single static page. Host it anywhere (Vercel, Netlify, GitHub Pages) or just open it in a browser.
- **GPU backend:** `colab/GOLDTECH_Voice.ipynb` — runs the model on Colab/Kaggle/Lightning/SageMaker and exposes it through a free Cloudflare tunnel. Auto-detects the platform.

```
Your browser (index.html)
      │  paste the tunnel URL once per session
      ▼
https://xxxx.trycloudflare.com   ← free Cloudflare tunnel
      ▼
FastAPI inside the Colab notebook  →  Chatterbox Turbo on the free T4 GPU
      ▼
WAV + MP3 voiceover in your voice
```

## Free GPU options (the notebook runs on all of these)

The notebook auto-detects the platform. Pick whichever gives you access:

| Platform | Free allowance | GPU | Notes |
|---|---|---|---|
| **Kaggle** ⭐ | ~30 hrs/week (reset weekly) | T4 x2 / P100 | Most reliable free quota. **Turn Internet ON** in Settings. Closest drop-in. |
| **Google Colab** | Unpublished, throttles | T4 (varies) | What you already use. Disconnects often; re-run last cell for a new URL. |
| **Lightning AI Studio** | 15 credits/month | T4 / L4 / A10G | Persistent storage, VS Code-like. Phone verification, no card. Good if Kaggle blocks tunnels. |
| **SageMaker Studio Lab** | ~4 GPU hrs/day | T4 | Persistent 15 GB, no credit card. Needs account approval. |
| **Hugging Face ZeroGPU** | 5 min/day | shared | Gradio-only — **not a fit** for this FastAPI+tunnel engine. |

**Recommendation:** try **Kaggle** first (best free quota, same notebook). If its tunnel is
blocked, use **Lightning AI Studio**. See the platform notes at the bottom of the notebook.

## Quick start

### 1. Start the engine
1. Open `colab/GOLDTECH_Voice.ipynb` on your chosen platform:
   - **Colab:** [colab.research.google.com](https://colab.research.google.com/) → File → Upload notebook.
   - **Kaggle:** [kaggle.com](https://www.kaggle.com/) → Create → Notebook → File → Import Notebook.
   - **Lightning / SageMaker:** upload the `.ipynb` into a GPU runtime.
2. Enable the GPU:
   - **Colab:** Runtime → Change runtime type → **T4 GPU**.
   - **Kaggle:** Settings → Accelerator → **GPU T4 x2**, and **Internet ON**.
3. **Run all** cells. (Colab only: approve the Drive prompt to cache the model — optional.)
4. The last cell prints a URL like `https://abc-xyz.trycloudflare.com`. **Copy it.**

### 2. Open the web app
- Open `index.html` in your browser (or deploy it — see below).
- Click **Connect engine** (top right), paste the URL, **Test connection**, **Save**.

### 3. Clone your voice
1. **Your voice sample:** press **Record** and speak naturally for **15–20 seconds** (quiet room, no music), or **Upload** a WAV/MP3 of yourself.
2. **Script:** paste what you want your voice to say.
3. Tick the permission checkbox, then **Generate voiceover**.
4. Play it back and **Download WAV / MP3**.

## Deploy the web app (optional)
It's one static file, so any static host works:
- **Vercel:** put `index.html` in a repo and import it (Framework preset: *Other*).
- **Netlify:** drag the folder into the Netlify dashboard.
- **GitHub Pages:** push `index.html` to a repo and enable Pages.

The app stores the engine URL in your browser's `localStorage`. The URL changes every
Colab session, so re-paste it when you restart the notebook.

## Settings that affect accuracy
| Setting | Default | Effect |
|---|---|---|
| Expressiveness (temperature) | 0.8 | Lower (0.5–0.7) = steadier, more faithful to the sample. Higher (1.0+) = more dramatic, less predictable. |
| Pause between chunks | 0.35s | Silence inserted between script chunks. |
| Chunk size | 220 chars | Smaller chunks keep the voice more consistent on long scripts. |
| Seed | random | Set a number to reproduce the same take. |

The engine conditions on your reference **once** and reuses it for every chunk, so the
voice stays identical across the whole script. Output is loudness-normalized to −16 LUFS.

## Tips for the closest clone
- Use **15–20s** of clean, natural speech — varied tone, no background noise, no echo.
- Match the **language/accent** you want the output in (Chatterbox Turbo is English-focused).
- Avoid whispering or shouting in the sample; speak at the volume you want the output.
- If a take sounds off, lower Expressiveness and try again, or record a cleaner sample.

## Files
```
goldtech-voice/
├── index.html                  # the web app (deploy this)
├── README.md
└── colab/
    ├── GOLDTECH_Voice.ipynb    # ← open this in Colab (the GPU engine)
    ├── server.py               # engine source (embedded in the notebook)
    ├── make_notebook.py        # regenerates the .ipynb from server.py
    └── test_server.py          # local API tests (stubbed model, no GPU needed)
```

## Notes & limits
- **Only clone a voice you own or have permission to use.** The app requires a consent tick, and Chatterbox embeds an inaudible neural watermark in every output.
- Free sessions sleep after a while and the tunnel URL changes on restart — just re-run the last notebook cell and re-paste the URL. This applies to Colab, Kaggle, Lightning, and SageMaker alike.
- One job runs at a time (the GPU is serialized). Cancel a running job from the app if needed.
- Chatterbox Turbo is English-focused. For other languages you'd swap in Chatterbox Multilingual — ask and I can add a language selector.

## License
The web app and notebook glue here are yours to use. Chatterbox Turbo is MIT-licensed
(© Resemble AI) — free for personal and commercial use. Keep its license notice if you
redistribute the model code.
