"""Tests for the app's Shape/Text mode toggle and the text-mode flow (all fakes, no webcam)."""

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from test_gestures import make_hand

from air_drawing import config
from air_drawing.app import (
    MSG_NO_TEXT,
    MSG_NOTHING_DRAWN,
    MSG_TEXT_NEEDS_GEMINI,
    TEXT_CONFIDENCE_LABEL,
    AirDrawingApp,
)
from air_drawing.classifier import Prediction
from air_drawing.hand_tracker import HandResult
from air_drawing.llm_fallback import TEXT_PROMPT, GeminiFallback
from air_drawing.ui import (
    ACTION_CLEAR,
    ACTION_MODE,
    RECOGNITION_SHAPE,
    RECOGNITION_TEXT,
    SOURCE_CNN,
    SOURCE_GEMINI,
    Toolbar,
)

# --- Fakes ----------------------------------------------------------------------------


class FakeTracker:
    def detect(self, frame: np.ndarray) -> None:
        return None

    def close(self) -> None:
        pass


class FakeClassifier:
    """Always predicts the same labels and counts calls."""

    def __init__(self, label: str = "circle", confidence: float = 0.95) -> None:
        self.calls = 0
        self._preds = [Prediction(label, confidence), Prediction("sun", 1 - confidence)]

    def predict(self, image: torch.Tensor) -> list[Prediction]:
        self.calls += 1
        return list(self._preds)


class FakeSpeaker:
    """Records sentences instead of speaking."""

    def __init__(self) -> None:
        self.muted = False
        self.sentences: list[str] = []

    def say(self, text: str, key: object = None) -> bool:
        self.sentences.append(text)
        return True

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def close(self) -> None:
        pass


class FakeGeminiClient:
    """Mimics `client.models.generate_content(...)` and records the requests."""

    def __init__(self, reply: dict | None = None, error: Exception | None = None) -> None:
        self.models = self
        self._reply = reply
        self._error = error
        self.requests: list[dict] = []

    def generate_content(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        return type("Response", (), {"text": json.dumps(self._reply)})()


# --- Helpers ----------------------------------------------------------------------------


def make_app(
    client: FakeGeminiClient | None = None, classifier: FakeClassifier | None = None
) -> tuple[AirDrawingApp, FakeSpeaker, FakeClassifier]:
    fallback = (
        GeminiFallback(model="test-model", client=client)
        if client is not None
        else GeminiFallback(api_key="", model="")
    )
    classifier = classifier or FakeClassifier()
    speaker = FakeSpeaker()
    app = AirDrawingApp(
        FakeTracker(),  # type: ignore[arg-type]
        classifier,  # type: ignore[arg-type]
        fallback,
        threshold=0.6,
        speaker=speaker,  # type: ignore[arg-type]
    )
    return app, speaker, classifier


def draw_word(app: AirDrawingApp) -> None:
    """Draw three short vertical strokes side by side (a wide, word-like drawing)."""
    for x in (100, 200, 300):
        for y in range(100, 200, 10):
            app.canvas.add_point((x, y))
        app.canvas.pen_up()


def wait_for_gemini(app: AirDrawingApp, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while app._pending is not None and time.monotonic() < deadline:
        app.poll_llm()
        time.sleep(0.01)
    assert app._pending is None, "Gemini call did not finish"


@pytest.fixture
def apps() -> Iterator[list[AirDrawingApp]]:
    created: list[AirDrawingApp] = []
    yield created
    for app in created:
        if app._fallback is not None:
            app._fallback.close()


# --- Mode toggle ----------------------------------------------------------------------


def test_starts_in_shape_mode() -> None:
    app, _, _ = make_app()
    assert app.recognition == RECOGNITION_SHAPE
    assert not app.text_mode


def test_key_t_toggles_mode() -> None:
    app, _, _ = make_app()
    assert app.handle_key(ord(config.KEY_TEXT_MODE))
    assert app.recognition == RECOGNITION_TEXT
    app.handle_key(ord(config.KEY_TEXT_MODE))
    assert app.recognition == RECOGNITION_SHAPE


def test_toolbar_mode_button_toggles_mode() -> None:
    app, _, _ = make_app()
    button = next(b for b in Toolbar(config.UI_DESIGN_WIDTH).buttons if b.action == ACTION_MODE)
    assert button.label == "Shape / Text"
    app.press(button)
    assert app.text_mode
    app.press(button)
    assert not app.text_mode


def test_existing_keys_still_work_in_text_mode() -> None:
    app, _, _ = make_app()
    app.handle_key(ord(config.KEY_TEXT_MODE))
    draw_word(app)
    app.handle_key(ord(config.KEY_UNDO))
    assert len(app.canvas.strokes) == 2
    app.handle_key(ord(config.KEY_CLEAR))
    assert app.canvas.is_empty
    assert app.text_mode
    assert not app.handle_key(ord(config.KEY_QUIT))


def test_toggle_discards_previous_result_and_keeps_drawing() -> None:
    app, _, _ = make_app()
    draw_word(app)
    app.predict()
    assert app.state.cnn
    app.toggle_recognition()
    assert not app.state.cnn
    assert len(app.canvas.strokes) == 3


def test_switching_to_text_mode_without_gemini_warns() -> None:
    app, _, _ = make_app()
    app.toggle_recognition()
    assert app.state.message == MSG_TEXT_NEEDS_GEMINI


# --- Text mode flow ---------------------------------------------------------------------


def test_text_mode_reads_handwriting_with_gemini(apps: list[AirDrawingApp]) -> None:
    client = FakeGeminiClient({"text": "  hello   world ", "confidence": 0.9})
    app, speaker, classifier = make_app(client)
    apps.append(app)
    app.toggle_recognition()
    draw_word(app)
    app.predict()
    wait_for_gemini(app)

    card = app.state.card()
    assert card.title == "hello world"
    assert card.source == SOURCE_GEMINI
    assert card.bars == [(TEXT_CONFIDENCE_LABEL, 0.9)]
    assert classifier.calls == 0  # the CNN is skipped in text mode
    assert speaker.sentences == ["You wrote hello world"]
    request = client.requests[0]
    assert request["model"] == "test-model"
    assert TEXT_PROMPT in request["contents"]


def test_text_mode_sends_wide_readable_image(apps: list[AirDrawingApp]) -> None:
    client = FakeGeminiClient({"text": "ok"})
    app, _, _ = make_app(client)
    apps.append(app)
    app.toggle_recognition()
    draw_word(app)
    app.predict()
    wait_for_gemini(app)
    png = client.requests[0]["contents"][0]
    image = cv2.imdecode(np.frombuffer(png.inline_data.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    assert width == config.TEXT_IMAGE_LONG_SIDE
    assert width > height  # aspect ratio of the word is kept
    assert image[0, 0] == 255 and image.min() == 0


def test_text_mode_without_gemini_shows_message(apps: list[AirDrawingApp]) -> None:
    app, speaker, classifier = make_app()
    apps.append(app)
    app.toggle_recognition()
    draw_word(app)
    app.predict()
    assert app.state.message == MSG_TEXT_NEEDS_GEMINI
    assert app._pending is None
    assert classifier.calls == 0
    assert speaker.sentences == []


def test_text_mode_gemini_error_shows_message(apps: list[AirDrawingApp]) -> None:
    app, speaker, _ = make_app(FakeGeminiClient(error=RuntimeError("quota exceeded")))
    apps.append(app)
    app.toggle_recognition()
    draw_word(app)
    app.predict()
    wait_for_gemini(app)
    assert app.state.message == MSG_TEXT_NEEDS_GEMINI
    assert app.state.card().title is None
    assert speaker.sentences == []


def test_text_mode_unreadable_text(apps: list[AirDrawingApp]) -> None:
    app, speaker, _ = make_app(FakeGeminiClient({"text": "", "confidence": 0.1}))
    apps.append(app)
    app.toggle_recognition()
    draw_word(app)
    app.predict()
    wait_for_gemini(app)
    assert app.state.message == MSG_NO_TEXT
    assert speaker.sentences == [config.SPEECH_UNKNOWN]


def test_text_mode_empty_canvas(apps: list[AirDrawingApp]) -> None:
    client = FakeGeminiClient({"text": "x"})
    app, _, _ = make_app(client)
    apps.append(app)
    app.toggle_recognition()
    app.predict()
    assert app.state.message == MSG_NOTHING_DRAWN
    assert client.requests == []


def test_text_result_is_not_repeated_for_same_drawing(apps: list[AirDrawingApp]) -> None:
    app, speaker, _ = make_app(FakeGeminiClient({"text": "hi"}))
    apps.append(app)
    app.toggle_recognition()
    draw_word(app)
    for _ in range(2):
        app.predict()
        wait_for_gemini(app)
    assert speaker.sentences == ["You wrote hi"]


# --- Shape mode is unchanged ----------------------------------------------------------


def test_shape_mode_uses_cnn_and_speaks_varied_phrase() -> None:
    app, speaker, classifier = make_app()
    draw_word(app)
    app.predict()
    assert classifier.calls == 1
    assert app.state.source == SOURCE_CNN
    assert app.state.card().title == "circle"
    assert len(speaker.sentences) == 1 and "circle" in speaker.sentences[0].lower()


def test_shape_mode_same_drawing_is_spoken_once() -> None:
    app, speaker, _ = make_app()
    draw_word(app)
    for _ in range(3):
        app.predict()
    assert len(speaker.sentences) == 1


def test_shape_mode_new_drawings_never_repeat_phrase_twice_in_a_row() -> None:
    app, speaker, _ = make_app()
    for i in range(12):
        app.canvas.add_point((i, i))  # each new point makes it a new drawing
        app.predict()
    assert len(speaker.sentences) == 12
    assert all(a != b for a, b in zip(speaker.sentences, speaker.sentences[1:], strict=False))


def test_process_frame_renders_in_both_modes() -> None:
    app, _, _ = make_app()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert app.process_frame(frame).shape == frame.shape
    app.toggle_recognition()
    assert app.process_frame(frame).shape == frame.shape


# --- Debug view, clear hold and Gemini hints -----------------------------------------


class RecordingClassifier(FakeClassifier):
    """Also keeps the last tensor it was given."""

    def __init__(self, label: str = "circle", confidence: float = 0.95) -> None:
        super().__init__(label, confidence)
        self.last_image: torch.Tensor | None = None

    def predict(self, image: torch.Tensor) -> list[Prediction]:
        self.last_image = image
        return super().predict(image)


def debug_app(
    tmp_path: Path,
    clock: Callable[[], float] = time.monotonic,
    classifier: FakeClassifier | None = None,
    client: FakeGeminiClient | None = None,
    threshold: float = 0.6,
) -> AirDrawingApp:
    fallback = (
        GeminiFallback(model="m", client=client)
        if client is not None
        else GeminiFallback(api_key="", model="")
    )
    return AirDrawingApp(
        FakeTracker(),  # type: ignore[arg-type]
        classifier or RecordingClassifier(),  # type: ignore[arg-type]
        fallback,
        threshold=threshold,
        speaker=FakeSpeaker(),  # type: ignore[arg-type]
        clock=clock,
        debug_dir=tmp_path,
    )


def test_key_d_toggles_debug() -> None:
    app, _, _ = make_app()
    assert not app.debug
    assert app.handle_key(ord(config.KEY_DEBUG))
    assert app.debug
    app.handle_key(ord(config.KEY_DEBUG))
    assert not app.debug


def test_debug_saves_exact_cnn_input_on_predict(tmp_path: Path) -> None:
    classifier = RecordingClassifier()
    app = debug_app(tmp_path, classifier=classifier)
    app.toggle_debug()
    draw_word(app)
    app.predict()
    files = list(tmp_path.glob("debug_*.png"))
    assert files == [app.last_debug_path]
    saved = cv2.imdecode(np.frombuffer(files[0].read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert app.debug_bitmap is not None and np.array_equal(saved, app.debug_bitmap)
    assert classifier.last_image is not None
    sent = (classifier.last_image[0, 0] * config.PIXEL_MAX).round().byte().numpy()
    assert np.array_equal(sent, saved)  # the saved file is exactly what the CNN received


def test_debug_saves_one_file_per_prediction(tmp_path: Path) -> None:
    app = debug_app(tmp_path)
    app.toggle_debug()
    draw_word(app)
    app.predict()
    app.canvas.add_point((500, 500))
    app.predict()
    assert len(list(tmp_path.glob("debug_*.png"))) == 2


def test_no_debug_files_when_debug_is_off(tmp_path: Path) -> None:
    app = debug_app(tmp_path)
    draw_word(app)
    app.predict()
    assert list(tmp_path.iterdir()) == []
    assert app.debug_bitmap is None


def test_debug_save_failure_does_not_crash(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    app = debug_app(blocker)  # saving into a file path fails
    app.toggle_debug()
    draw_word(app)
    app.predict()
    assert app.state.cnn  # prediction still happened
    assert app.debug_bitmap is not None


def test_debug_view_is_drawn(tmp_path: Path) -> None:
    app = debug_app(tmp_path)
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    off = app.process_frame(blank.copy())
    app.toggle_debug()
    on = app.process_frame(blank.copy())
    corner = (slice(400, 700), slice(1050, 1280))
    assert on[corner].sum() > off[corner].sum()


def hand_frames(app: AirDrawingApp, hand: HandResult, count: int, now: list[float]) -> None:
    for _ in range(count):
        now[0] += FRAME_DT
        app.update(hand)


FRAME_DT = 1 / 30


def open_hand() -> HandResult:
    return HandResult(make_hand(True, True, True, True, True) + 300, "Right")


def index_hand(x: float = 0.0) -> HandResult:
    return HandResult(make_hand(index=True) + np.array([300 + x, 300.0]), "Right")


def test_open_hand_must_be_held_before_clearing(tmp_path: Path) -> None:
    now = [0.0]
    app = debug_app(tmp_path, clock=lambda: now[0])
    for i in range(20):
        hand_frames(app, index_hand(i * 5), 1, now)
    assert not app.canvas.is_empty
    hold_frames = int(config.CLEAR_HOLD_SECONDS / FRAME_DT)
    hand_frames(app, open_hand(), hold_frames // 2, now)
    assert not app.canvas.is_empty
    assert 0.0 < app.clear_progress < 1.0  # ring is filling
    hand_frames(app, open_hand(), hold_frames, now)
    assert app.canvas.is_empty
    assert app.clear_progress == 0.0


def test_brief_open_hand_while_drawing_does_not_clear(tmp_path: Path) -> None:
    now = [0.0]
    app = debug_app(tmp_path, clock=lambda: now[0])
    for i in range(20):
        hand_frames(app, index_hand(i * 5), 1, now)
    hand_frames(app, open_hand(), int(0.5 / FRAME_DT), now)  # half a second
    hand_frames(app, index_hand(200), 10, now)
    hand_frames(app, open_hand(), int(0.5 / FRAME_DT), now)  # timer restarted
    assert not app.canvas.is_empty


def test_clear_key_and_button_still_clear_immediately(tmp_path: Path) -> None:
    app = debug_app(tmp_path)
    draw_word(app)
    app.handle_key(ord(config.KEY_CLEAR))
    assert app.canvas.is_empty
    draw_word(app)
    clear = next(b for b in Toolbar(config.UI_DESIGN_WIDTH).buttons if b.action == ACTION_CLEAR)
    app.press(clear)
    assert app.canvas.is_empty


def test_low_confidence_sends_cnn_top3_as_hints(tmp_path: Path) -> None:
    client = FakeGeminiClient({"label": "bicycle", "confidence": 0.8})
    app = debug_app(
        tmp_path,
        classifier=RecordingClassifier("car", 0.55),
        client=client,
        threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
    )
    try:
        draw_word(app)
        app.predict()
        wait_for_gemini(app)
        prompt = client.requests[0]["contents"][1]
        assert "car (55%), sun (45%)" in prompt
        assert all(c in prompt for c in config.CATEGORIES)
        assert app.state.card().title == "bicycle"  # Gemini chose a non-hinted category
        assert app.state.source == SOURCE_GEMINI
    finally:
        app._fallback.close()  # type: ignore[union-attr]


def test_default_threshold_is_raised() -> None:
    assert config.DEFAULT_CONFIDENCE_THRESHOLD == 0.85
    example = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "CONFIDENCE_THRESHOLD=0.85" in example.splitlines()
