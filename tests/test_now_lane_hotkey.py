"""The pause hotkey while a click plays through a hold: it stops that read, at any stage of it.

From the review of 8fbfc25 (JMO, D112 B): a press during the click's synthesis released every
room in whole-message mode, and a click a recording paused never resumed, so two presses did
nothing at all. Every press must stop the read or release; never neither.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

import claude_code_tts.daemon as d
import claude_code_tts.state as st
from claude_code_tts.bridge import JOBS
from claude_code_tts.state import hold
from tests import test_now_lane as nl


@pytest.mark.parametrize("unit", ["message", "sentence"])
def test_a_press_during_the_clicks_synthesis_keeps_the_hold(tmp_path: Path, unit: str) -> None:
    state_dir, queue_dir, spoken, stop = nl._run_loop(tmp_path, "sleep 0.5", {"speech_unit": unit})

    def slow(text, persona, output_file, **kw):
        spoken.append(text)
        if "clicked" in text:
            time.sleep(1.0)
        with wave.open(str(output_file), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)
        return True

    p = patch.object(d, "daemon_generate_speech", side_effect=slow)
    p.start()
    try:
        hold([])
        time.sleep(0.05)
        nl._enqueue_room(queue_dir, "a room waits", "alice--claude--a")
        click = nl._click("what JMO clicked")
        assert nl._wait_for(lambda: JOBS.state(click["id"]) == "synthesizing")
        nl._pause_hotkey()
        state = st.read_playback_state()
        assert state.get("paused") is True, f"the press released every room: {state}"
        assert nl._wait_for(lambda: JOBS.state(click["id"]) == "cancelled")
        time.sleep(0.3)
        assert "a room waits" not in spoken
    finally:
        p.stop()
        stop()


def test_a_click_a_recording_paused_resumes_and_a_press_stops_it(tmp_path: Path) -> None:
    playing = tmp_path / "click-playing"
    body = f'case "$*" in *browser*) echo x >> {playing}; sleep 3 ;; *) sleep 0.05 ;; esac'
    state_dir, queue_dir, spoken, stop = nl._run_loop(tmp_path, body)

    def long_click(text, persona, output_file, **kw):
        # 6 s, so the resume is far from the near-end skip and really plays.
        spoken.append(text)
        with wave.open(str(output_file), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 48000)
        return True

    p = patch.object(d, "daemon_generate_speech", side_effect=long_click)
    p.start()
    try:
        hold([])
        time.sleep(0.05)
        click = nl._click("what JMO clicked")
        assert nl._wait_for(lambda: playing.exists())
        st.write_playback_state(mic_held=True)
        assert nl._wait_for(lambda: JOBS.state(click["id"]) == "paused")
        st.write_playback_state(mic_held=False)
        assert nl._wait_for(lambda: len(playing.read_text().splitlines()) == 2), "the click resumes"
        nl._pause_hotkey()
        assert nl._wait_for(lambda: JOBS.state(click["id"]) == "cancelled")
        state = st.read_playback_state()
        assert state["paused"] is True and "stop_read" not in state
    finally:
        p.stop()
        stop()


def test_a_stop_for_the_read_on_deck_at_start_drops_it(tmp_path: Path) -> None:
    loop_harness, stop_loop = nl._harness()
    state_dir, queue_dir, spoken, patches = loop_harness(tmp_path, {})
    for p in patches:
        p.start()
    try:
        on_deck = {"id": "j1", "session_id": "browser", "lane": "now", "project": "hear:x", "text": "stopped",
                   "audio_position": 0.5, "timestamp": time.time()}
        st.write_playback_state(paused=False, audio_pid=None, current_message=on_deck, stop_read="j1")
        import threading

        def run() -> None:
            d._shutdown_requested = False
            d._daemon_mode = True
            d.daemon_loop()

        runner = threading.Thread(target=run, daemon=True)
        runner.start()
        try:
            assert nl._wait_for(lambda: (state_dir / "daemon.heartbeat").exists())
            time.sleep(0.3)
            assert spoken == []
            state = st.read_playback_state()
            assert not state.get("current_message") and "stop_read" not in state
        finally:
            stop_loop(runner)
    finally:
        for p in reversed(patches):
            p.stop()
