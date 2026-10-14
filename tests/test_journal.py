import json

import pytest

from codeagent.journal import Conflict
from codeagent.plan import TaskStatus
from codeagent.policy import Violation
from codeagent.records import Mode, RunStatus
from codeagent.runtime import InvalidRequest

from .support import FIXED_ORDER, ROOT, Agent, SimulatedCrash, call, final, reply

ORIGINAL_ORDER = """def calculate_total(prices: list[float], discount_percent: float) -> float:
    return sum(prices) - discount_percent
"""


def edit_script() -> list[dict]:
    return [
        reply(
            call("c1", "write_file", path="order.py", content=FIXED_ORDER),
            call("c2", "write_file", path="helpers.py", content="RATE = 100\n"),
        ),
        final(),
    ]


def test_diff_shows_what_the_run_changed(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.start()
    diff = agent.runtime.diff(agent.run_id)
    assert "--- a/order.py\n+++ b/order.py\n" in diff
    assert "-    return sum(prices) - discount_percent\n" in diff
    assert "+    return sum(prices) * (100 - discount_percent) / 100\n" in diff
    assert "--- /dev/null\n+++ b/helpers.py\n@@ -0,0 +1 @@\n+RATE = 100\n" in diff


def test_each_action_stores_its_own_diff_and_hashes(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.start()
    changes = {change.path: change for change in agent.store.file_changes(agent.run_id)}
    assert changes["helpers.py"].original is None
    assert changes["order.py"].original != changes["order.py"].final
    write = agent.store.calls(agent.run_id)[0]
    assert write.changes == ("order.py",)
    attempt = agent.store.attempts(write)[0]
    assert attempt.diff.startswith("--- a/order.py\n+++ b/order.py\n")
    assert "helpers.py" not in attempt.diff


def test_rollback_after_restart_restores_only_the_runs_files(tmp_path):
    agent = Agent(tmp_path, edit_script())
    (agent.workspace / "notes.txt").write_text("mine")
    agent.start()
    (agent.workspace / "notes.txt").write_text("mine, edited later")

    agent.restart()
    restored = agent.runtime.rollback(agent.run_id)

    assert restored == ("helpers.py", "order.py")
    assert agent.file("order.py") == ORIGINAL_ORDER
    assert not (agent.workspace / "helpers.py").exists()
    assert agent.file("notes.txt") == "mine, edited later"
    assert agent.run.status is RunStatus.ROLLED_BACK
    with pytest.raises(InvalidRequest, match="already rolled back"):
        agent.runtime.rollback(agent.run_id)


def test_rollback_restores_a_deleted_file(tmp_path):
    agent = Agent(tmp_path, [reply(call("c1", "delete_file", path="order.py")), final()])
    agent.start()
    agent.runtime.decide_action(agent.run_id, approve=True)
    assert not (agent.workspace / "order.py").exists()
    agent.runtime.rollback(agent.run_id)
    assert agent.file("order.py") == ORIGINAL_ORDER


def test_external_edit_makes_rollback_refuse_to_write_anything(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.start()
    (agent.workspace / "order.py").write_text("someone else's fix\n")

    with pytest.raises(Conflict) as conflict:
        agent.runtime.rollback(agent.run_id)

    assert conflict.value.paths == ("order.py",)
    assert agent.file("order.py") == "someone else's fix\n"
    assert agent.file("helpers.py") == "RATE = 100\n"
    assert agent.run.status is RunStatus.COMPLETED


def test_rollback_rechecks_paths_before_writing(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(
                call("c1", "write_file", path="order.py", content=FIXED_ORDER),
                call("c2", "write_file", path="pkg/helper.py", content="RATE = 100\n"),
            ),
            final(),
        ],
    )
    agent.start()
    elsewhere = tmp_path / "elsewhere"
    (agent.workspace / "pkg").rename(elsewhere)
    (agent.workspace / "pkg").symlink_to(elsewhere)
    agent.restart()
    with pytest.raises(Violation, match="symlinks are not allowed"):
        agent.runtime.rollback(agent.run_id)
    assert agent.file("order.py") == FIXED_ORDER
    assert (elsewhere / "helper.py").exists()


def test_rollback_cut_short_by_a_crash_can_be_run_again(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.start()
    (agent.workspace / "order.py").write_text(ORIGINAL_ORDER)
    restored = agent.runtime.rollback(agent.run_id)
    assert restored == ("helpers.py", "order.py")
    assert not (agent.workspace / "helpers.py").exists()


def test_action_that_writes_into_a_denied_directory_stops_the_run_and_can_be_undone(tmp_path):
    sneaky_test = "import pathlib\npathlib.Path('private').mkdir()\npathlib.Path('private/out.txt').write_text('x')\n"
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "write_file", path="test_zz.py", content=sneaky_test)),
            reply(call("c2", "bash", command="python3 -m unittest -v", cwd=".")),
            final(),
        ],
    )
    run = agent.start()
    assert run.status is RunStatus.FAILED
    assert "policy does not allow: private/out.txt" in run.note
    agent.runtime.rollback(agent.run_id)
    assert not (agent.workspace / "private" / "out.txt").exists()
    assert not (agent.workspace / "test_zz.py").exists()


def test_drift_before_an_action_blocks_it_until_the_workspace_is_consistent(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.start(mode=Mode.SAFE)
    tests = agent.file("test_order.py")
    (agent.workspace / "test_order.py").write_text("tampered\n")

    run = agent.runtime.decide_action(agent.run_id, approve=True)

    assert run.status is RunStatus.BLOCKED
    assert "test_order.py" in run.note
    assert agent.file("order.py") == ORIGINAL_ORDER
    (agent.workspace / "test_order.py").write_text(tests)
    run = agent.runtime.resume(agent.run_id)
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert agent.file("order.py") == FIXED_ORDER


def test_bash_changes_to_project_files_are_journaled(tmp_path):
    agent = Agent(
        tmp_path,
        [reply(call("c1", "bash", command="touch generated.txt", cwd=".")), final()],
        allow_commands=("touch generated.txt",),
    )
    agent.start()
    assert [change.path for change in agent.store.file_changes(agent.run_id)] == ["generated.txt"]
    agent.runtime.rollback(agent.run_id)
    assert not (agent.workspace / "generated.txt").exists()


def test_rollback_waits_until_an_interrupted_call_is_resolved(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.crash_after("write_file")
    with pytest.raises(SimulatedCrash):
        agent.start()
    agent.heal()
    with pytest.raises(InvalidRequest, match="resolve the interrupted call"):
        agent.runtime.rollback(agent.run_id)
    assert agent.file("order.py") == FIXED_ORDER


def test_rollback_answers_a_call_still_waiting_for_approval(tmp_path):
    agent = Agent(tmp_path, edit_script())
    agent.start(mode=Mode.SAFE)
    agent.runtime.rollback(agent.run_id)
    assert agent.run.status is RunStatus.ROLLED_BACK
    assert "rolled back" in agent.tool_messages()[0]["content"]
    assert agent.store.open_calls(agent.run_id) == []


def test_an_action_that_edits_the_policy_file_is_undone_and_stops_the_run(tmp_path):
    evil = "import pathlib\npathlib.Path('../policy.toml').write_text('[commands]\\nallow = [\"rm -rf .\"]\\n')\n"
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "write_file", path="test_evil.py", content=evil)),
            reply(call("c2", "bash", command="python3 -m unittest -v", cwd=".")),
            final(),
        ],
    )
    policy = agent.policy_path.read_text()
    run = agent.start()
    assert run.status is RunStatus.FAILED
    assert "modified the policy file" in run.note
    assert agent.policy_path.read_text() == policy


def test_rollback_survives_a_rename_that_differs_only_in_letter_case(tmp_path):
    rename = "import os\nos.rename('NOTES.txt', 'notes.txt')\n"
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "write_file", path="test_rename.py", content=rename)),
            reply(call("c2", "bash", command="python3 -m unittest -v", cwd=".")),
            final(),
        ],
    )
    (agent.workspace / "NOTES.txt").write_text("keep me")
    agent.restart()
    agent.start()
    agent.runtime.rollback(agent.run_id)
    names = sorted(path.name for path in agent.workspace.iterdir() if path.name.lower() == "notes.txt")
    assert names == ["NOTES.txt"]
    assert agent.file("NOTES.txt") == "keep me"


def test_a_breach_fails_the_active_task_and_answers_the_rest_of_the_batch(tmp_path):
    steps = json.loads((ROOT / "examples" / "fix_order_plan.json").read_text())
    sneaky = "import pathlib\npathlib.Path('private').mkdir()\npathlib.Path('private/out.txt').write_text('x')\n"
    agent = Agent(
        tmp_path,
        [
            *steps[:2],
            reply(
                call("t1", "write_file", path="test_zz.py", content=sneaky),
                call("t2", "bash", command="python3 -m unittest -v", cwd="."),
                call("t3", "read_file", path="order.py"),
            ),
        ],
    )
    agent.start(plan_first=True)
    run = agent.runtime.decide_plan(agent.run_id, approve=True)
    assert run.status is RunStatus.FAILED
    assert agent.store.tasks(agent.run_id)[0].status is TaskStatus.FAILED
    assert agent.store.open_calls(agent.run_id) == []
