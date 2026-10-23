"""Offline text-to-speech (pyttsx3) in a background thread fed by a queue.

The video loop only ever puts sentences on a queue, so speech never blocks it.
A fresh pyttsx3 engine is created per sentence because reusing one engine makes
`runAndWait()` hang on the second call on Windows (SAPI5).
"""

import logging
import queue
import threading
from collections.abc import Callable, Hashable
from typing import Any

from air_drawing import config

logger = logging.getLogger(__name__)

EngineFactory = Callable[[], Any]

_STOP = object()  # sentinel that ends the worker thread


def make_sentence(label: str) -> str:
    """Return the sentence to speak for a predicted label."""
    if label == config.OTHER_LABEL:
        return config.SPEECH_UNKNOWN
    article = "an" if label[:1].lower() in config.VOWELS else "a"
    return config.SPEECH_TEMPLATE.format(article=article, label=label)


def _default_engine_factory() -> Any:
    """Create a pyttsx3 engine (imported lazily so tests never need audio)."""
    import pyttsx3

    return pyttsx3.init()


def _init_thread_com() -> None:
    """Initialise COM for this thread on Windows; no-op elsewhere."""
    try:
        import pythoncom
    except ImportError:
        return
    pythoncom.CoInitialize()


class Speaker:
    """Speaks sentences one at a time on a daemon thread."""

    def __init__(
        self,
        enabled: bool = config.VOICE_ENABLED,
        rate: int = config.VOICE_RATE,
        volume: float = config.VOICE_VOLUME,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        """Start the worker; `engine_factory` can be injected (e.g. a fake in tests)."""
        self.muted = not enabled
        self._rate = rate
        self._volume = volume
        self._factory = engine_factory or _default_engine_factory
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.SPEECH_QUEUE_SIZE)
        self._last: tuple[Hashable, str] | None = None
        self._thread = threading.Thread(target=self._worker, name="speech", daemon=True)
        self._thread.start()

    def say(self, text: str, key: Hashable = None) -> bool:
        """Queue `text` without blocking; return True if it was queued.

        Skipped when muted, or when the same sentence was already queued for the
        same `key` (e.g. the same drawing).
        """
        if self.muted or not self._thread.is_alive():
            return False
        if key is not None and self._last == (key, text):
            return False
        self._last = (key, text) if key is not None else None
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            self._drop_oldest()
            self._queue.put_nowait(text)
        return True

    def toggle_mute(self) -> bool:
        """Flip mute; muting also drops sentences still waiting. Returns the new state."""
        self.muted = not self.muted
        if self.muted:
            self._drain()
        logger.info("Voice %s", "muted" if self.muted else "unmuted")
        return self.muted

    def wait_idle(self) -> None:
        """Block until every queued sentence has been spoken (used by tests)."""
        self._queue.join()

    def close(self) -> None:
        """Stop the worker thread, dropping pending sentences."""
        self._drain()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=config.SPEECH_JOIN_TIMEOUT_S)

    def _worker(self) -> None:
        """Speak queued sentences until the stop sentinel arrives."""
        _init_thread_com()
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._speak(str(item))
            finally:
                self._queue.task_done()

    def _speak(self, text: str) -> None:
        """Speak one sentence with a fresh engine; errors are logged, never raised."""
        try:
            engine = self._factory()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:  # missing audio device, driver errors...
            logger.warning("Text-to-speech failed: %s", exc)
            return
        logger.info("Spoke: %s", text)

    def _drop_oldest(self) -> None:
        """Remove one waiting sentence to make room."""
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass

    def _drain(self) -> None:
        """Remove every waiting sentence."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            self._queue.task_done()
