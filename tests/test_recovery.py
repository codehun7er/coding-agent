import pytest

from codeagent.records import AttemptStatus, CallStatus, RunStatus
from codeagent.runtime import InvalidRequest, Resolution

from .support import FIXED_ORDER, TEST_COMMAND, Agent, SimulatedCrash, call, final, reply


def fix_script() -> list[dict]:
    return [
        reply(call("c1", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("c2", "bash", command=TEST_COMMAND, cwd=".")),
        final(),
    ]


def crashed_agent(tmp_path, tool_name: str, steps: list[dict] | None = None) -> Agent:
    agent = Agent(tmp_path, steps or fix_script())
    agent.crash_after(tool_name)
    with pytest.raises(SimulatedCrash):
        agent.start()
    agent.heal()
    return agent


def only_call(agent: Agent, name: str):
    return next(call for call in agent.store.calls(agent.run_id) if call.name == name)


def test_side_effect_is_marked_started_before_it_runs(tmp_path):
    agent = crashed_agent(tmp_path, "write_file")
    write = only_call(agent, "write_file")
    assert write.status is CallStatus.STARTED
    assert [attempt.status for attempt in agent.store.attempts(write)] == [AttemptStatus.STARTED]
    assert agent.file("order.py") == FIXED_ORDER


def test_interrupted_write_waits_for_the_user_and_shows_detected_changes(tmp_path):
    agent = crashed_agent(tmp_path, "write_file")
    run = agent.runtime.advance(agent.run_id)
    assert run.status is RunStatus.UNCERTAIN
    interruption = agent.runtime.interruption(agent.run_id)
    assert interruption.call.name == "write_file"
    assert interruption.drift.paths == ("order.py",)
    assert "+    if not 0 <= discount_percent <= 100:" in interruption.drift.text
    assert interruption.options == (Resolution.RETRY, Resolution.FAIL, Resolution.ABORT)
    assert agent.runtime.advance(agent.run_id).status is RunStatus.UNCERTAIN


def test_resolution_must_say_who_owns_detected_changes(tmp_path):
    agent = crashed_agent(tmp_path, "write_file")
    agent.runtime.advance(agent.run_id)
    with pytest.raises(InvalidRequest, match="--adopt"):
        agent.runtime.resolve(agent.run_id, Resolution.RETRY)


def test_retry_is_a_new_attempt_of_the_same_call(tmp_path):
    agent = crashed_agent(tmp_path, "write_file")
    agent.runtime.advance(agent.run_id)
    run = agent.runtime.resolve(agent.run_id, Resolution.RETRY, adopt=True)

    assert run.status is RunStatus.COMPLETED
    write = only_call(agent, "write_file")
    assert [attempt.status for attempt in agent.store.attempts(write)] == [
        AttemptStatus.INTERRUPTED,
        AttemptStatus.FINISHED,
    ]
    assert write.changes == ("order.py",)
    assert [change.path for change in agent.store.file_changes(agent.run_id)] == ["order.py"]
    assert run.used.side_effects == 3


def test_bash_is_never_retried(tmp_path):
    agent = crashed_agent(tmp_path, "bash")
    agent.runtime.advance(agent.run_id)
    assert Resolution.RETRY not in agent.runtime.interruption(agent.run_id).options
    with pytest.raises(InvalidRequest, match="not safe to repeat"):
        agent.runtime.resolve(agent.run_id, Resolution.RETRY)


def test_failed_resolution_reports_the_unknown_outcome_to_the_model(tmp_path):
    agent = crashed_agent(tmp_path, "bash")
    agent.runtime.advance(agent.run_id)
    run = agent.runtime.resolve(agent.run_id, Resolution.FAIL)
    assert run.status is RunStatus.COMPLETED
    bash_result = next(m for m in agent.tool_messages() if m["tool_call_id"] == "c2")
    assert "outcome is unknown" in bash_result["content"]
    assert len([attempt for attempt in agent.store.attempts(only_call(agent, "bash"))]) == 1


def test_abort_ends_the_run_with_an_error(tmp_path):
    agent = crashed_agent(tmp_path, "bash")
    agent.runtime.advance(agent.run_id)
    run = agent.runtime.resolve(agent.run_id, Resolution.ABORT)
    assert run.status is RunStatus.FAILED
    assert [d["verdict"] for d in agent.store.decisions(agent.run_id)] == ["abort"]


def test_disowned_changes_stay_out_of_the_journal(tmp_path):
    agent = crashed_agent(tmp_path, "write_file")
    agent.runtime.advance(agent.run_id)
    agent.runtime.resolve(agent.run_id, Resolution.FAIL, adopt=False)
    assert agent.store.file_changes(agent.run_id) == []
    assert agent.runtime.rollback(agent.run_id) == ()
    assert agent.file("order.py") == FIXED_ORDER


def test_interrupted_read_is_repeated_without_asking(tmp_path):
    agent = crashed_agent(tmp_path, "read_file", [reply(call("c1", "read_file", path="order.py")), final()])
    run = agent.runtime.advance(agent.run_id)
    assert run.status is RunStatus.COMPLETED
    read = only_call(agent, "read_file")
    assert [attempt.status for attempt in agent.store.attempts(read)] == [
        AttemptStatus.INTERRUPTED,
        AttemptStatus.FINISHED,
    ]


def test_repeated_tool_call_id_returns_the_stored_result_without_running_again(tmp_path):
    write = call("c1", "write_file", path="order.py", content=FIXED_ORDER)
    agent = Agent(tmp_path, [reply(write), reply(write), final()])
    agent.start()
    writes = agent.store.calls(agent.run_id)
    assert [executed.name for executed in agent.reporter.executed] == ["write_file"]
    assert agent.store.attempts(writes[1]) == []
    assert writes[0].result == writes[1].result


def test_reused_tool_call_id_with_other_arguments_is_a_protocol_error(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "read_file", path="order.py")),
            reply(call("c1", "write_file", path="order.py", content="broken")),
            final(),
        ],
    )
    agent.start()
    assert "protocol error" in agent.tool_messages()[1]["content"]
    assert "broken" not in agent.file("order.py")


def test_malformed_unicode_in_arguments_is_an_argument_error(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(
                {"id": "c1", "name": "read_file", "arguments": '{"path": "x\\ud800"}'},
                {"id": "c2", "name": "write_file", "arguments": '{"path": "a.txt", "content": "\\ud800"}'},
            ),
            final(),
        ],
    )
    run = agent.start()
    assert run.status is RunStatus.COMPLETED
    assert all("not valid Unicode" in message["content"] for message in agent.tool_messages())
    assert not (agent.workspace / "a.txt").exists()


def test_a_reply_with_unstorable_text_does_not_jam_the_run(tmp_path):
    agent = Agent(tmp_path, [{"content": "done \ud800"}])
    run = agent.start()
    assert run.status is RunStatus.COMPLETED
    assert run.note == "done ?"
