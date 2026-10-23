# Air Drawing Recognition

Draw shapes in the air with your index finger in front of a webcam. MediaPipe tracks
your hand, OpenCV renders the drawing, and a small CNN trained on Google's
[Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset recognises it. When the
CNN is unsure, the drawing is sent to Gemini as a fallback.

> Work in progress. Full setup instructions will follow.

## Setup

_TODO_

## Usage

_TODO_

## Design decisions

- **`pyproject.toml` (tool config only).** Not in the original file list. It configures
  `ruff` and tells `pytest` to put `src/` on the import path, so tests can import
  `air_drawing` without packaging or `sys.path` hacks in every test file.

## Known issues

_None yet._
