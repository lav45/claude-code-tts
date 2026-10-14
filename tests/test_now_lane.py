"""Lane "now": a block a person clicked to hear plays first, cutting off a room mid-message.

The room's message resumes where it was cut once no now block is left (JMO 2026-10-13: "a
click on hear is where his eyes are, so it speaks FIRST"). A person's hold set before the
click does not hold the click; a pause pressed during the read, and a recording, do.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import claude_code_tts.daemon as d
import claude_code_tts.msgqueue as mq
import claude_code_tts.state as st
from claude_code_tts.bridge import JOBS, write_bridge_message
from claude_code_tts.state import hold, may_play


def _m(ts: float, **extra: object) -> dict:
    return {"timestamp": ts, "text": f"t{ts}", "_file": Path(f"{ts}.json"), **extra}


def test_now_goes_before_everything_and_keeps_its_own_order() -> None:
    msgs = [_m(1.0, type="control"), _m(2.0, lane="background"), _m(3.0), _m(4.0, lane="now"), _m(5.0, lane="now")]
    assert [x["timestamp"] for x in mq.play_order(msgs)] == [4.0, 5.0, 1.0, 3.0, 2.0]


class TestMayPlay:
    held = {"paused": True, "paused_by": "user", "paused_since": 100.0}

    def test_a_click_after_a_persons_hold_plays(self) -> None:
        assert may_play(self.held, "browser", "now", 101.0)

    def test_a_pause_pressed_during_the_read_holds_it(self) -> None:
        assert not may_play(self.held, "browser", "now", 99.0)

    def test_a_recording_holds_it(self) -> None:
        assert not may_play({**self.held, "paused_by": "mic"}, "browser", "now", 101.0)
        assert not may_play({**self.held, "mic_held": True}, "browser", "now", 101.0)

    def test_other_lanes_and_a_missing_stamp_wait_as_before(self) -> None:
        assert not may_play(self.held, "browser", "background", 101.0)
        assert not may_play(self.held, "browser", "now", None)
        assert not may_play(self.held, "browser", "now", True)


def _harness():
    spec = importlib.util.spec_from_file_location("test_pause_ledger", Path(__file__).parent / "test_pause_ledger.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.loop_harness, mod.stop_loop


def _wait_for(pred, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _enqueue_room(queue_dir: Path, text: str, session_id: str) -> None:
    ts = time.time()
    msg_id = f"{abs(hash(text)) % 10**8:08x}"
    f = queue_dir / f"{ts:.6f}_{msg_id}.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"id": msg_id, "session_id": session_id, "project": session_id, "text": text, "timestamp": ts}))
    tmp.rename(f)


def _click(text: str) -> dict:
    """What the bridge does for POST /speak with lane "now"."""
    msg = write_bridge_message(text, persona="claude-prime", persona_config={}, source="hear", lane="now")
    d.request_preempt()
    return msg


def _run_loop(tmp_path: Path, player_body: str, raw_config: dict | None = None):
    loop_harness, stop_loop = _harness()
    state_dir, queue_dir, spoken, patches = loop_harness(tmp_path, raw_config or {})
    player = tmp_path / "fake-player"
    player.write_text(f"#!/bin/bash\n{player_body}\n")
    player.chmod(0o755)
    def generate(text, persona, output_file, **kw):
        # Room speech is 6 s of WAV, so a cut early in it is far from the near-end skip
        # (NEAR_END_THRESHOLD) and the resume really plays; a click is a short block.
        spoken.append(text)
        import wave

        with wave.open(str(output_file), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * (800 if "clicked" in text else 48000))
        return True

    patches.append(patch.object(d, "daemon_generate_speech", side_effect=generate))
    for p in patches:
        p.start()
    JOBS._jobs.clear()
    JOBS._cancel.clear()
    st.write_playback_state(paused=False, paused_by=None, audio_pid=None, current_message=None)
    d._PREEMPT["at"] = 0.0

    def run() -> None:
        d._shutdown_requested = False
        d._daemon_mode = True
        d.daemon_loop()

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    assert _wait_for(lambda: (state_dir / "daemon.heartbeat").exists()), "loop never reached its idle pass"

    def stop() -> None:
        stop_loop(runner)
        for p in reversed(patches):
            p.stop()
        d._PREEMPT["at"] = 0.0

    return state_dir, queue_dir, spoken, stop


@pytest.mark.parametrize("unit", ["message", "sentence"])
def test_a_click_cuts_a_room_off_and_the_room_resumes_after(tmp_path: Path, unit: str) -> None:
    playing = tmp_path / "room-playing"
    # The room's audio runs long; the page's is short. The WAV name carries the session.
    body = f'case "$*" in *browser*) sleep 0.05 ;; *) echo "$*" >> {playing}; sleep 1 ;; esac'
    state_dir, queue_dir, spoken, stop = _run_loop(tmp_path, body, {"speech_unit": unit})
    try:
        _enqueue_room(queue_dir, "a long room reply", "alice--claude--a")
        assert _wait_for(lambda: playing.exists()), "the room never started playing"
        started = time.monotonic()
        _click("what JMO clicked")
        assert _wait_for(lambda: "what JMO clicked" in spoken)
        assert time.monotonic() - started < 1.5, "the click waited for the room's message to end"
        assert _wait_for(lambda: len(playing.read_text().splitlines()) == 2), "the room's message never played again"
        assert spoken == ["a long room reply", "what JMO clicked", "a long room reply"]
        assert not st.read_playback_state().get("preempted_message")
    finally:
        stop()
    log = (state_dir / "daemon.log").read_text()
    assert "Skipping replay" not in log
    if unit == "sentence":
        assert "Message interrupted at sentence 1/1" in log
    else:
        assert "Replaying interrupted message from start" in log or "Resuming from" in log
    assert "Cut off alice--claude--a for a now block" in log
    assert "Resuming alice--claude--a, cut off by a now block" in log
    assert d._PREEMPT["at"] == 0.0


def test_a_click_plays_through_a_hold_set_before_it(tmp_path: Path) -> None:
    state_dir, queue_dir, spoken, stop = _run_loop(tmp_path, "sleep 0.05")
    try:
        hold([])
        time.sleep(0.05)
        _enqueue_room(queue_dir, "a room waits", "alice--claude--a")
        _click("what JMO clicked")
        assert _wait_for(lambda: "what JMO clicked" in spoken)
        time.sleep(0.3)
        assert spoken == ["what JMO clicked"], "the room must wait for the hold"
        assert st.read_playback_state()["paused"] is True
    finally:
        stop()
