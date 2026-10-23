"""Tests for the speech queue (fake pyttsx3 engine, no real audio)."""

import threading
import time
from collections.abc import Iterator

import pytest

from air_drawing import config
from air_drawing.speech import Speaker, make_sentence


class FakeEngine:
    """Records what would have been spoken."""

    def __init__(self, log: list[str], delay: float = 0.0, fail: bool = False) -> None:
        self._log = log
        self._delay = delay
        self._fail = fail
        self.properties: dict[str, object] = {}
        self._pending: list[str] = []

    def setProperty(self, name: str, value: object) -> None:  # noqa: N802 (pyttsx3 API)
        self.properties[name] = value

    def say(self, text: str) -> None:
        self._pending.append(text)

    def runAndWait(self) -> None:  # noqa: N802 (pyttsx3 API)
        if self._fail:
            raise RuntimeError("no audio device")
        time.sleep(self._delay)
        self._log.extend(self._pending)
        self._pending.clear()

    def stop(self) -> None:
        pass


@pytest.fixture
def spoken() -> list[str]:
    return []


@pytest.fixture
def speaker(spoken: list[str]) -> Iterator[Speaker]:
    s = Speaker(enabled=True, engine_factory=lambda: FakeEngine(spoken))
    yield s
    s.close()


@pytest.mark.parametrize(
    ("label", "sentence"),
    [
        ("triangle", "I think it's a triangle"),
        ("apple", "I think it's an apple"),
        ("umbrella", "I think it's an umbrella"),
        (config.OTHER_LABEL, config.SPEECH_UNKNOWN),
    ],
)
def test_make_sentence(label: str, sentence: str) -> None:
    assert make_sentence(label) == sentence


def test_say_speaks_in_background(speaker: Speaker, spoken: list[str]) -> None:
    assert speaker.say("hello", key=1)
    speaker.wait_idle()
    assert spoken == ["hello"]


def test_same_sentence_for_same_drawing_is_not_repeated(
    speaker: Speaker, spoken: list[str]
) -> None:
    assert speaker.say("I think it's a cat", key=7)
    assert not speaker.say("I think it's a cat", key=7)
    assert speaker.say("I think it's a cat", key=8)  # new drawing
    assert speaker.say("I think it's a sun", key=8)  # new answer for the same drawing
    speaker.wait_idle()
    assert spoken == ["I think it's a cat", "I think it's a cat", "I think it's a sun"]


def test_mute_blocks_and_unmute_restores(speaker: Speaker, spoken: list[str]) -> None:
    assert speaker.toggle_mute() is True
    assert not speaker.say("quiet", key=1)
    assert speaker.toggle_mute() is False
    assert speaker.say("loud", key=1)
    speaker.wait_idle()
    assert spoken == ["loud"]


def test_disabled_in_config_starts_muted(spoken: list[str]) -> None:
    s = Speaker(enabled=False, engine_factory=lambda: FakeEngine(spoken))
    try:
        assert s.muted
        assert not s.say("nothing")
    finally:
        s.close()


def test_say_does_not_block_on_slow_engine(spoken: list[str]) -> None:
    s = Speaker(enabled=True, engine_factory=lambda: FakeEngine(spoken, delay=0.3))
    try:
        start = time.perf_counter()
        s.say("one", key=1)
        s.say("two", key=2)
        assert time.perf_counter() - start < 0.1
        s.wait_idle()
        assert spoken == ["one", "two"]
    finally:
        s.close()


def test_rate_and_volume_are_applied() -> None:
    engines: list[FakeEngine] = []

    def factory() -> FakeEngine:
        engines.append(FakeEngine([]))
        return engines[-1]

    s = Speaker(enabled=True, rate=123, volume=0.4, engine_factory=factory)
    try:
        s.say("x")
        s.wait_idle()
        assert engines[0].properties == {"rate": 123, "volume": 0.4}
    finally:
        s.close()


def test_engine_errors_do_not_kill_the_worker(spoken: list[str]) -> None:
    calls = {"n": 0}

    def factory() -> FakeEngine:
        calls["n"] += 1
        return FakeEngine(spoken, fail=calls["n"] == 1)

    s = Speaker(enabled=True, engine_factory=factory)
    try:
        s.say("fails", key=1)
        s.say("works", key=2)
        s.wait_idle()
        assert spoken == ["works"]
    finally:
        s.close()


def test_full_queue_keeps_newest(spoken: list[str]) -> None:
    gate = threading.Event()

    class BlockingEngine(FakeEngine):
        def runAndWait(self) -> None:  # noqa: N802
            gate.wait(timeout=5)
            super().runAndWait()

    s = Speaker(enabled=True, engine_factory=lambda: BlockingEngine(spoken))
    try:
        s.say("first", key=0)
        time.sleep(0.1)  # let the worker pick up "first" and block on the gate
        for i in range(1, config.SPEECH_QUEUE_SIZE + 3):
            assert s.say(f"s{i}", key=i)
        gate.set()
        s.wait_idle()
        assert spoken[0] == "first"
        assert spoken[-1] == f"s{config.SPEECH_QUEUE_SIZE + 2}"
        assert len(spoken) == 1 + config.SPEECH_QUEUE_SIZE
    finally:
        s.close()


def test_close_stops_thread(spoken: list[str]) -> None:
    s = Speaker(enabled=True, engine_factory=lambda: FakeEngine(spoken))
    s.close()
    assert not s.say("after close")
