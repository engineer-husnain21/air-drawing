# CLAUDE.md — Air Drawing Recognition

## 0. Rules
- This file is the only source of truth for this project.
- Do NOT create any file or folder outside section 3. If one is truly unavoidable,
  add it, update section 3 of this file, and explain why under "Design decisions" in README.md.
- Build all phases (section 9) in order without waiting for me between phases.
  Stop and ask ONLY for something you cannot do yourself (e.g. GitHub login).
- Do NOT add any dependency outside section 2 without a strong reason (record it in README).

## 1. Project summary
A Python desktop app where the user draws shapes in the air with their index finger
in front of a webcam. MediaPipe tracks the hand, OpenCV renders the drawing, and a
small CNN trained on Google's Quick, Draw! dataset recognises the drawing. If the
model's confidence is low, the drawing is sent to Gemini (vision LLM) as a fallback.

## 2. Tech stack
- Python 3.12
- mediapipe (Tasks API, Hand Landmarker)
- opencv-python
- torch, torchvision
- numpy, requests
- google-genai (Gemini fallback)
- python-dotenv
- pyttsx3 (offline text-to-speech)
- pytest, ruff (dev)

## 3. Folder structure (STRICT, complete list)
```
air-drawing/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── pyproject.toml              # Tool config only: ruff + pytest (src/ on path)
├── main.py                     # Entry point: runs the live app
├── data/
│   ├── raw/                    # Quick, Draw! .npy files (gitignored)
│   └── samples/                # A few demo PNG drawings (committed)
├── models/                     # hand_landmarker.task + cnn_quickdraw.pt (gitignored)
├── scripts/
│   ├── download_data.py        # Downloads .npy files + hand_landmarker.task
│   └── train.py                # Trains the CNN, saves weights to models/
├── src/
│   └── air_drawing/
│       ├── __init__.py
│       ├── config.py           # All constants: categories, thresholds, paths
│       ├── hand_tracker.py     # MediaPipe wrapper, returns 21 landmarks
│       ├── gestures.py         # Detects draw / pen-up / clear / predict
│       ├── canvas.py           # Stroke storage, smoothing, rendering
│       ├── preprocess.py       # Canvas -> 28x28 tensor in Quick, Draw! format
│       ├── model.py            # CNN architecture
│       ├── classifier.py       # Loads weights, returns top-3 predictions
│       ├── llm_fallback.py     # Gemini: shape fallback + handwriting reading
│       ├── speech.py           # Speaks predictions (pyttsx3, background queue)
│       ├── ui.py               # Toolbar, air-click dwell, prediction card, badges
│       └── app.py              # Main loop tying everything together
└── tests/
    ├── test_app.py             # Mode toggle + text-mode flow (fakes only)
    ├── test_preprocess.py
    ├── test_gestures.py
    ├── test_canvas.py
    ├── test_classifier.py
    ├── test_speech.py
    └── test_ui.py
```

## 4. Categories (15)
circle, triangle, square, star, house, car, tree, fish, sun, cloud,
apple, umbrella, cat, clock, bicycle
Defined only in `config.py`. Everything else imports it from there.

## 5. Features
- Mirror the webcam frame horizontally. Landmark #8 (index fingertip) is the pen.
- Gestures (same gesture for ~5 consecutive frames before acting = debounce):
  - Only index finger up → DRAW
  - Index + middle up → PEN UP
  - All 5 fingers up → CLEAR
  - Fist → PREDICT
- Keyboard fallback: `c` clear, `p` predict, `q` quit.
- Exponential smoothing on fingertip points.
- On screen: hand landmarks, current gesture/mode, top-3 predictions with %,
  and a "via CNN" / "via Gemini" label.
- Voice output: after each prediction speak "I think it's a <label>" (Gemini label if it
  answered). Background thread + queue, no repeat for the same drawing, mute with `m`
  or the toolbar. Settings (enabled, rate, volume) in `config.py`.
- Toolbar: rounded buttons Clear, Predict, Undo, Mute/Unmute, Quit, a colour palette
  (6+ colours) and 2-3 pen thicknesses. Click with the mouse or "air click" (hold the
  fingertip on a button ~1 s, progress ring shown). No drawing over the toolbar.
  Pen colour/thickness are display-only; preprocessing uses the raw stroke points.
- Prediction card (large top-1, top-3 confidence bars, source) and a coloured mode badge
  (Draw / Pen up / Idle). Extra keys: `u` undo, `m` mute.
- Shape / Text mode: toggle with `t` or the toolbar button "Shape / Text"; a badge shows
  the mode. Shape mode = CNN first, Gemini below threshold. Text mode skips the CNN and
  sends a readable black-on-white image (aspect ratio kept, fixed stroke width) to Gemini,
  which returns JSON `{"text", "confidence"}`; the card shows the text large "via Gemini".
  Without Gemini show "Text mode needs Gemini" (never crash).
- Speech phrases: shape mode picks randomly from `SHAPE_PHRASES` in `config.py`, never the
  same phrase twice in a row (keep a/an and the "I'm not sure" phrase for "other");
  text mode says "You wrote {text}".

## 6. Preprocessing (critical)
Quick, Draw! bitmaps are 28x28, black background (0), white strokes (255).
`preprocess.py` must: crop to bounding box with padding → pad to square keeping aspect
ratio → center → draw strokes thick enough to survive resizing → resize to 28x28 →
scale to [0, 1]. Training and inference MUST use the same scaling.
Empty canvas → skip prediction and show "Nothing drawn".

## 7. Model and training
- Small CNN: 2–3 conv blocks + dropout + linear head.
- `download_data.py`: downloads each category's numpy bitmap `.npy` into `data/raw/`,
  and the MediaPipe `hand_landmarker.task` into `models/`. Skip files that already exist.
- `train.py`: CLI args `--samples-per-class` (default 15000), `--epochs` (default 8),
  `--categories` (default all). 80/20 split, log accuracy per epoch, save best weights
  to `models/cnn_quickdraw.pt`. Must run on CPU.

## 8. Gemini fallback
- Only when top-1 confidence < `CONFIDENCE_THRESHOLD` (default 0.6, from `.env`).
- Ask Gemini to choose one label (from the categories, or "other") and return JSON.
- Read `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env`. Never hardcode keys or model names.
- Run the call in a background thread so the webcam loop never freezes.
- If the key is missing or the call fails, keep the CNN result and show
  "LLM fallback unavailable". The app must never crash because of the LLM.

## 9. Phases (in order; finish and verify one before the next)
1. Setup: structure, requirements, .gitignore, .env.example, config.py, README skeleton
2. hand_tracker.py + webcam loop showing landmarks (main.py, app.py)
3. gestures.py + canvas.py (drawing, pen up, clear) + tests
4. model.py + download_data.py + train.py
5. preprocess.py + classifier.py + live prediction + tests
6. llm_fallback.py + integration into app
7. README final (setup, download, train, run, controls, demo GIF placeholder)

## 10. Coding rules
- All comments, docstrings, logs and terminal output in English.
- Type hints on all functions; short docstrings.
- No magic numbers: put them in `config.py`. Use `pathlib` for paths.
- Use the `logging` module, not `print()` (except CLI progress in scripts is fine via logging).
- One responsibility per module, as in section 3.
- Tests never use the webcam, real network, or a real LLM (use fakes/mocks).

## 11. Done check (after every phase)
- `ruff check .`, `ruff format --check .` and `pytest` pass.
- No file outside section 3, no hardcoded keys/paths/model names, no `print()`.

## 12. Git and GitHub
- Before Phase 1: run `gh auth status`. If not logged in, STOP and ask me to run
  `gh auth login`. Never ask for or handle tokens or passwords.
- If no remote exists: `gh repo create air-drawing --public --source=. --remote=origin`.
  If the repo already exists on GitHub, just add it as `origin`.
- Branch: `main`.
- Small commits: one logical unit each (3–6 commits per phase), only when checks pass.
- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.
- Push after every commit: `git push -u origin main`.
- Before every commit run `git status`; never stage `.env`, `.venv/`, `data/raw/`,
  `models/*.task`, `models/*.pt` or cache files.

## 13. Final report (only at the very end)
- What was built, GitHub repo URL, total commits.
- Exact commands for me to: create venv, install, set up `.env`, download data, train, run.
- Design decisions and known issues.
