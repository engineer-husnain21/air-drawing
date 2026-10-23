# Air Drawing Recognition

Draw shapes in the air with your index finger in front of a webcam. MediaPipe tracks
your hand, OpenCV renders the drawing, and a small CNN trained on Google's
[Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset recognises it. When the
CNN is unsure, the drawing is sent to Gemini (vision LLM) as a fallback.

![Demo](docs-demo-placeholder.gif)
<!-- TODO: record a demo GIF and replace the placeholder above. -->

**Recognised categories (15):** circle, triangle, square, star, house, car, tree, fish,
sun, cloud, apple, umbrella, cat, clock, bicycle.

## How it works

```
webcam ─► mirror ─► MediaPipe Hand Landmarker ─► gesture (debounced) ─► canvas strokes
                                                                            │ fist / "p"
                                                                            ▼
                         Gemini (background thread) ◄─ conf < threshold ◄─ CNN top-3
```

1. Each frame is mirrored; landmark #8 (index fingertip) is the pen.
2. Fingertip points are smoothed (exponential moving average) and stored as strokes.
3. On *predict*, strokes are converted to the Quick, Draw! format: crop to bounding
   box → square with padding (aspect ratio kept) → centre → thick strokes at 256×256 →
   resize to 28×28 → scale to [0, 1]. Training uses the exact same scaling function.
4. The CNN returns the top-3 labels. If the top-1 confidence is below
   `CONFIDENCE_THRESHOLD`, the drawing is sent to Gemini without blocking the video.

## Setup

Requires **Python 3.12** and a webcam.

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env      # then edit .env
```

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then edit .env
```

`requirements.txt` holds the runtime dependencies; `requirements-dev.txt` adds `pytest`
and `ruff`. For a smaller, CPU-only PyTorch, install torch first with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

### `.env`

| Variable               | Meaning                                                  |
| ---------------------- | -------------------------------------------------------- |
| `GEMINI_API_KEY`       | Your key from <https://aistudio.google.com/apikey>.     |
| `GEMINI_MODEL`         | Any Gemini model with vision, e.g. `gemini-2.5-flash`.   |
| `CONFIDENCE_THRESHOLD` | Ask Gemini below this top-1 CNN confidence (default 0.6).|

The app works without a key: it keeps the CNN result and shows
"LLM fallback unavailable".

## Download data

```bash
python scripts/download_data.py                              # all 15 categories + hand model
python scripts/download_data.py --categories circle square   # just some categories
```

Downloads each category's numpy bitmap file (~70–130 MB each, ~1.5 GB total) into
`data/raw/` and `hand_landmarker.task` into `models/`. Existing files are skipped.

## Train

```bash
python scripts/train.py                                   # 15,000 samples/class, 8 epochs, all categories
python scripts/train.py --samples-per-class 5000 --epochs 4
python scripts/train.py --categories circle triangle square --epochs 3
```

Runs on CPU. Uses an 80/20 train/validation split, logs loss and accuracy per epoch
and saves the best weights (with the category list) to `models/cnn_quickdraw.pt`.
The full default run needs about 1 GB of RAM.

## Run

```bash
python main.py
```

## Controls

### Gestures

A gesture must be held for ~5 consecutive frames before it is applied.

| Gesture                      | Action        |
| ---------------------------- | ------------- |
| Only index finger up         | Draw          |
| Index + middle finger up     | Pen up (move) |
| All 5 fingers up (open hand) | Clear canvas  |
| Fist                         | Predict       |

### Toolbar

The toolbar at the top of the window has rounded buttons **Clear**, **Predict**,
**Undo** (remove the last stroke), **Mute/Unmute**, **Quit** and **Shape / Text**, a
palette of 8 pen colours and 3 pen thicknesses. Each button works two ways:

- **Mouse:** left-click it.
- **Air click:** hold your fingertip over it for ~1 second. A progress ring fills around
  the fingertip; the click fires when it is full. Move off the button to click it again.

Nothing is drawn while the fingertip is over the toolbar. Pen colour and thickness only
change how strokes look on screen. Recognition always uses the raw stroke points, so
they do not affect accuracy.

### Keyboard

| Key | Action                 |
| --- | ---------------------- |
| `c` | Clear canvas           |
| `p` | Predict                |
| `u` | Undo last stroke       |
| `m` | Mute / unmute voice    |
| `t` | Shape / Text mode      |
| `q` | Quit                   |

### Shape and Text mode

Press `t` or the **Shape / Text** button to switch. A badge (top right) shows the mode;
the button turns purple in Text mode.

- **Shape mode** (default): the CNN recognises one of the 15 shapes; below
  `CONFIDENCE_THRESHOLD` the drawing goes to Gemini as a fallback.
- **Text mode:** write letters, digits or words, then predict (fist, `p` or the
  **Predict** button). The CNN is skipped and Gemini reads the handwriting directly. The
  card shows the text in large letters with "via Gemini" and Gemini's confidence. Text
  mode needs `GEMINI_API_KEY` and `GEMINI_MODEL`; without them the card shows
  "Text mode needs Gemini" and the app keeps running.

Tip: lift the pen (index + middle finger) between letters and words. The window is
mirrored, so your writing reads the right way round.

### On screen

- **Prediction card** (top left): the top-1 label in large text, the CNN top-3 as coloured
  confidence bars, and "via CNN" or "via Gemini". If Gemini answered, its label is the
  large title. In Text mode the recognised text is the title (long text is shrunk, then
  shortened with "...").
- **Badges** (top right): Draw, Pen up or Idle; Shape mode or Text mode; and "Muted"
  when voice is off.

### Voice

After every prediction the app says the result. In Shape mode it picks a phrase at
random from `SHAPE_PHRASES` in `config.py` (*"That's a triangle"*, *"Looks like an
apple"*, *"I think it's a cat"*, *"star!"*), never the same phrase twice in a row, and
uses the Gemini label if Gemini answered. Unknown drawings get *"I'm not sure what that
is"*. In Text mode it says *"You wrote hello"*. It speaks once per drawing: predicting
the same drawing again stays silent. Voice settings (`VOICE_ENABLED`, `VOICE_RATE`,
`VOICE_VOLUME`) are in `src/air_drawing/config.py`. Speech is offline (`pyttsx3`,
SAPI5 on Windows) and runs on a background thread, so the video never pauses.

## Development

```bash
ruff check .
ruff format --check .
pytest
```

Tests never use the webcam, the network or a real LLM (Gemini is replaced by a fake
client).

## Project layout

```
main.py                     entry point
scripts/download_data.py    data + hand model download
scripts/train.py            CNN training
src/air_drawing/
  config.py                 all constants: categories, thresholds, paths
  hand_tracker.py           MediaPipe wrapper (21 landmarks)
  gestures.py               draw / pen up / clear / predict + debouncing
  canvas.py                 strokes, smoothing, rendering
  preprocess.py             strokes -> 28x28 Quick, Draw! tensor
  model.py                  CNN
  classifier.py             loads weights, top-3 predictions
  llm_fallback.py           Gemini: shape fallback + handwriting reading (background thread)
  speech.py                 text-to-speech queue (pyttsx3, background thread)
  ui.py                     toolbar, air click, prediction card, mode badge
  app.py                    main loop and HUD
tests/                      pytest suite
data/samples/               demo drawings
```

## Design decisions

- **`pyproject.toml` (tool config only).** Not in the original file list. It configures
  `ruff` and tells `pytest` to put `src/` on the import path, so tests can import
  `air_drawing` without packaging or `sys.path` hacks in every test file. `main.py`
  and the two scripts add `src/` to `sys.path` themselves, so no install step is needed.
- **Rotation-invariant gestures.** A finger counts as "up" when its tip is clearly
  farther from the wrist than its PIP joint (the thumb is measured against the pinky
  MCP). This works for tilted hands, unlike simple y-coordinate comparisons.
- **Debouncing with one-shot actions.** Clear and predict fire once, on the frame the
  stable gesture changes, not every frame the gesture is held. While drawing, points
  are only added on frames whose raw gesture is also "draw", so the finger curling into
  a fist does not leave a tail on the drawing.
- **Preprocessing calibrated on real data.** Padding (10 %) and stroke thickness
  (13 px at 256×256) were chosen so a drawn circle has the same ink density and
  framing as real Quick, Draw! circles.
- **Checkpoint stores its categories.** `cnn_quickdraw.pt` contains the category list,
  so a model trained on a subset (e.g. the smoke test) still loads and labels correctly.
- **Gemini answer format.** Gemini is asked for JSON `{"label", "confidence"}`.
  Unknown labels are mapped to `other`; any error (network, quota, bad JSON, missing
  key or model) is logged and the CNN result is kept.
- **Stale LLM answers are dropped.** Clearing or predicting again while a Gemini call
  is in flight discards the old answer.
- **Hand model loaded from memory.** `hand_landmarker.task` is passed to MediaPipe as
  bytes, which avoids MediaPipe path problems on Windows (spaces, non-ASCII).
- **One pyttsx3 engine per sentence.** On Windows, reusing a pyttsx3 engine makes the
  second `runAndWait()` hang (checked with pyttsx3 2.99). The speech thread therefore
  creates a fresh engine for each sentence (~0.2–0.6 s, off the video thread). The queue
  holds at most 4 sentences; when it is full the oldest is dropped.
- **"Same drawing" = canvas version.** The canvas bumps a version number on every
  added point, undo and clear. Speech skips a sentence it already spoke for the same
  version, so predicting twice stays quiet but a new answer (e.g. from Gemini) is spoken.
  When the CNN is unsure and Gemini is asked, the app waits for Gemini (or its failure)
  and speaks only once.
- **Display-only pen styles.** Each stroke stores its own colour and thickness next to
  its points; `Canvas.strokes` still returns only the points, so preprocessing is unchanged.
- **Scaled toolbar.** The toolbar is laid out for a 1280 px wide frame and scaled down
  uniformly for narrower cameras, so all buttons always fit.
- **Air click fires once.** After a dwell click the fingertip has to leave the button
  before it can click it again, so holding still does not repeat the click.
- **Separate image for handwriting.** The shape image is a 256×256 square, which would
  squash a word into a thin strip with blobby letters. Text mode renders a separate
  image: aspect ratio kept, 1024 px on the long side, and a fixed 8 px stroke in output
  pixels (independent of the on-screen pen thickness), so letters stay readable whether
  they were written large or small.
- **One Gemini client, two prompts.** `GeminiFallback` shares the request and error
  handling between shape classification (`submit`) and text reading (`submit_text`).
  Both run on the same background thread.
- **Speech de-duplication happens before a phrase is chosen.** With random phrases the
  sentence changes every time, so the app remembers the last (drawing version, mode,
  result) it spoke and stays silent if nothing changed.
- **Mode button placed after Quit.** It sits at the end of the action buttons, so
  existing button positions and tests did not change.
- **LLM tests live in `tests/test_classifier.py`.** The file list has no separate test
  file for `llm_fallback.py`, and the fallback is part of classification.

## Known issues

- **Not tested with a live webcam during development.** All modules are covered by unit
  tests and the full app loop was driven with synthetic hand landmarks, but the gesture
  thresholds (`FINGER_EXTENDED_RATIO`, `THUMB_EXTENDED_RATIO` in `config.py`) may need
  tuning for your camera and hand.
- **Only a smoke model was trained** (2 categories, 500 samples, 1 epoch). Run the full
  training before expecting good predictions.
- **Two OpenCV packages.** `mediapipe` depends on `opencv-contrib-python`, which gets
  installed next to `opencv-python`. Both provide `cv2` and currently work together; if
  `import cv2` ever breaks, uninstall both and reinstall only `opencv-contrib-python`.
- **Voice output was checked silently.** pyttsx3 was verified on this machine by
  rendering speech to WAV files from a background thread; audible playback and the
  live toolbar were not tested with a real webcam/speakers. If no speech engine is
  available, the app logs a warning and keeps running.
- **Non-ASCII text needs OpenCV 5.** OpenCV 5 (installed here) draws accented,
  Cyrillic and Chinese text on the card. With OpenCV 4.x, which `requirements.txt`
  still allows, those characters show as "?" (the spoken sentence is unaffected).
- **Text mode reliability depends on Gemini.** Air-written text is shaky; short words in
  block capitals work best.
- **Demo GIF** is still a placeholder.
