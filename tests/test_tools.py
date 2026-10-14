import time
from pathlib import Path

import pytest

from codeagent.policy import Violation
from codeagent.settings import load_settings
from codeagent.tools import Outcome, Toolbox, ToolFailure

from .support import TEST_COMMAND, make_project


@pytest.fixture
def toolbox(tmp_path: Path) -> Toolbox:
    workspace, policy_path = make_project(tmp_path, allow_commands=(TEST_COMMAND, "ls", "sleep 10", "ls missing"))
    (workspace / "private").mkdir()
    (workspace / "private" / "notes.txt").write_text("hidden")
    (workspace / "__pycache__").mkdir()
    return Toolbox(load_settings(policy_path, workspace))


def run(toolbox: Toolbox, name: str, **arguments) -> Outcome:
    return toolbox.get(name).execute(arguments, 10_000)


def test_bash_returns_stdout_stderr_and_exit_code(toolbox: Toolbox):
    outcome = run(toolbox, "bash", command=TEST_COMMAND, cwd=".")
    assert outcome.ok
    assert outcome.fields["exit_code"] == 1
    assert "FAILED" in outcome.streams["stderr"]
    assert outcome.streams["stdout"] == ""


def test_bash_reports_a_failing_command(toolbox: Toolbox):
    outcome = run(toolbox, "bash", command="ls missing", cwd=".")
    assert outcome.fields["exit_code"] != 0
    assert "missing" in outcome.streams["stderr"]


def test_bash_stops_at_the_timeout(toolbox: Toolbox):
    outcome = run(toolbox, "bash", command="sleep 10", cwd=".")
    assert not outcome.ok
    assert "timed out" in outcome.fields["error"]


def test_bash_refuses_commands_the_policy_refuses(toolbox: Toolbox):
    with pytest.raises(Violation):
        run(toolbox, "bash", command="ls; rm order.py", cwd=".")


def test_list_files_hides_denied_and_ignored_entries(toolbox: Toolbox):
    entries = run(toolbox, "list_files", path=".").streams["entries"].splitlines()
    assert entries == ["order.py", "test_order.py"]


def test_write_then_read_round_trips(toolbox: Toolbox):
    assert run(toolbox, "write_file", path="pkg/new.py", content="x = 1\n").fields["created"] is True
    assert run(toolbox, "read_file", path="pkg/new.py").streams["content"] == "x = 1\n"


def test_reading_a_missing_file_fails(toolbox: Toolbox):
    with pytest.raises(ToolFailure, match="no such file"):
        run(toolbox, "read_file", path="missing.py")


def test_clip_marks_output_as_incomplete():
    outcome = Outcome(True, {}, {"stdout": "abcdef", "stderr": "xyz"}).clip(4)
    assert outcome.streams == {"stdout": "abcd", "stderr": ""}
    assert outcome.size == 4
    assert '"truncated"' in outcome.render()


def test_write_ignores_a_planted_staging_symlink(toolbox: Toolbox, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    workspace = tmp_path / "workspace"
    (workspace / ".notes.txt.partial").symlink_to(outside)
    run(toolbox, "write_file", path="notes.txt", content="new")
    assert outside.read_text() == "untouched"
    assert (workspace / "notes.txt").read_text() == "new"
    assert not (workspace / "notes.txt").is_symlink()


def test_bash_returns_when_the_command_exits_even_if_a_detached_child_keeps_running(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path, allow_commands=("python3 spawn.py",))
    (workspace / "spawn.py").write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'], start_new_session=True)\n"
        "print('started')\n"
    )
    toolbox = Toolbox(load_settings(policy_path, workspace))
    began = time.monotonic()
    outcome = toolbox.get("bash").execute({"command": "python3 spawn.py", "cwd": "."}, 1000)
    assert time.monotonic() - began < 2
    assert outcome.fields["exit_code"] == 0
    assert outcome.streams["stdout"] == "started\n"


def test_bash_output_is_limited_to_what_the_caller_allows(toolbox: Toolbox):
    outcome = toolbox.get("bash").execute({"command": TEST_COMMAND, "cwd": "."}, 20)
    assert outcome.size <= 20
    assert outcome.truncated


def test_write_keeps_the_file_mode(toolbox: Toolbox, tmp_path: Path):
    script = tmp_path / "workspace" / "run.sh"
    script.write_text("old")
    script.chmod(0o755)
    run(toolbox, "write_file", path="run.sh", content="new")
    assert script.stat().st_mode & 0o777 == 0o755
