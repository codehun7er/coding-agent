import os
import subprocess
import sys
from pathlib import Path

from .support import FIXED_ORDER, ROOT, make_project


class Cli:
    def __init__(self, root: Path) -> None:
        self.workspace, self.policy = make_project(root)
        self.state = root / "state"

    def __call__(self, *arguments: str, crash_after: str | None = None) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(ROOT)}
        environment.pop("CODEAGENT_CRASH_AFTER", None)
        if crash_after:
            environment["CODEAGENT_CRASH_AFTER"] = crash_after
        return subprocess.run(
            [sys.executable, "-m", "codeagent", "--state", str(self.state), *arguments],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )

    def start(self, script: str, *options: str, crash_after: str | None = None) -> tuple[str, str]:
        result = self(
            "run",
            "Fix the order total so the tests pass without changing them",
            "--workspace",
            str(self.workspace),
            "--policy",
            str(self.policy),
            "--script",
            str(ROOT / "examples" / script),
            *options,
            crash_after=crash_after,
        )
        run_id = result.stdout.split()[1]
        return run_id, result.stdout


def test_plan_mode_session_from_the_command_line(tmp_path):
    cli = Cli(tmp_path)
    run_id, output = cli.start("fix_order_plan.json", "--plan", "--mode", "auto")
    assert f"run {run_id}: awaiting_plan" in output
    assert "T3. Run the test suite (after T2)" in output
    assert f"codeagent plan approve {run_id}" in output

    output = cli("plan", "approve", run_id).stdout
    assert "- [ ] T1 Read order.py to find how the total is computed  <- in progress" in output
    assert "- [x] T3 Run the test suite" in output
    assert f"run {run_id}: completed" in output

    tasks = cli("tasks", run_id).stdout
    assert tasks.splitlines()[:3] == [
        "- [x] T1 Read order.py to find how the total is computed",
        "- [x] T2 Rewrite calculate_total to apply a percentage discount and raise ValueError outside 0..100",
        "- [x] T3 Run the test suite",
    ]
    assert "T3: bash call t5 shows that" in tasks

    assert "+++ b/order.py" in cli("diff", run_id).stdout
    assert "restored: order.py" in cli("rollback", run_id).stdout
    assert "return sum(prices) - discount_percent" in (cli.workspace / "order.py").read_text()


def test_safe_mode_approve_and_deny_from_the_command_line(tmp_path):
    cli = Cli(tmp_path)
    run_id, output = cli.start("fix_order.json")
    assert "write_file" in output
    assert f"codeagent approve {run_id}" in output

    output = cli("approve", run_id).stdout
    assert f"run {run_id}: awaiting_approval" in output
    assert "bash" in output

    output = cli("deny", run_id, "--reason", "not now").stdout
    assert "error: the user denied this action: not now" in output
    assert f"run {run_id}: completed" in output
    assert (cli.workspace / "order.py").read_text() == FIXED_ORDER


def test_crash_and_recovery_from_the_command_line(tmp_path):
    cli = Cli(tmp_path)
    run_id, _ = cli.start("fix_order.json", "--mode", "auto", crash_after="write_file")
    assert (cli.workspace / "order.py").read_text() == FIXED_ORDER

    status = cli("resume", run_id).stdout
    assert f"run {run_id}: uncertain" in status
    assert "+++ b/order.py" in status
    assert f"codeagent resolve {run_id} {{retry|fail|abort}} --adopt|--disown" in status

    refused = cli("resolve", run_id, "retry")
    assert refused.returncode == 2
    assert "--adopt" in refused.stderr

    output = cli("resolve", run_id, "retry", "--adopt").stdout
    assert f"run {run_id}: completed" in output
    assert "side_effects 3/20" in output
