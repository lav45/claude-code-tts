"""An external player takes the daemon's audio through config, not through a shadowed paplay.

TTS Player (a desktop player with its own queue and history) used to put a script named paplay
first in the daemon's PATH. On a Mac the daemon looks for afplay first and never found it, and
the script read the current message id back from playback.json. config.json "player_command"
names the player, CLAUDE_TTS_MESSAGE_ID carries the id, and CLAUDE_TTS_QUEUE_DIR lets the
hook's `speak` hand the message to the player instead of the daemon.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import claude_code_tts.audio as audio
import claude_code_tts.daemon as d
from claude_code_tts.config import TTS_CONFIG_FILE


def write_config(raw: dict) -> None:
    TTS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TTS_CONFIG_FILE.write_text(json.dumps(raw))


def test_player_command_wins_over_detection(monkeypatch) -> None:
    write_config({"player_command": ["tts-player", "sink"]})
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert audio.detect_player() == ["tts-player", "sink"]


def test_malformed_player_command_falls_back_to_detection(monkeypatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/paplay" if name == "paplay" else None)
    for bad in ("tts-player sink", [], ["tts-player", 1], [""]):
        write_config({"player_command": bad})
        assert audio.detect_player() == ["paplay"]
    write_config({})
    assert audio.detect_player() == ["paplay"]


def test_player_env_names_the_current_message(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_TTS_MESSAGE_ID", "stale")
    assert d.player_env({"current_message": {"id": "abc123"}})["CLAUDE_TTS_MESSAGE_ID"] == "abc123"
    assert "CLAUDE_TTS_MESSAGE_ID" not in d.player_env({"current_message": None})
    assert "CLAUDE_TTS_MESSAGE_ID" not in d.player_env({})


def test_daemon_runs_the_player_with_the_message_id(tmp_path: Path) -> None:
    seen = tmp_path / "seen"
    player = tmp_path / "player"
    player.write_text(f'#!/bin/bash\necho "$CLAUDE_TTS_MESSAGE_ID ${{!#}}" > {seen}\n')
    player.chmod(0o755)
    wav = tmp_path / "s0.wav"
    wav.write_bytes(b"")
    state = {"paused": False, "audio_pid": None, "current_message": {"id": "abc123"}}
    with (
        patch.object(d, "detect_player", return_value=[str(player)]),
        patch.object(d, "read_playback_state", return_value=state),
        patch.object(d, "write_playback_state"),
    ):
        ok, killed, _ = d.daemon_play_audio(wav)
    assert (ok, killed) == (True, False)
    assert seen.read_text() == f"abc123 {wav}\n"


def test_queue_dir_follows_the_environment(tmp_path: Path) -> None:
    code = "from claude_code_tts import msgqueue; print(msgqueue.QUEUE_DIR)"
    env = {**os.environ, "CLAUDE_TTS_QUEUE_DIR": str(tmp_path / "hook-queue")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(tmp_path / "hook-queue")
    env.pop("CLAUDE_TTS_QUEUE_DIR")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(Path(os.environ["HOME"]) / ".claude-tts" / "queue")
