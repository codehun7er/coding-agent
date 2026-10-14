import hashlib

from codeagent.records import RunStatus

from .support import FIXED_ORDER, TEST_COMMAND, Agent, call, final, reply


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def control_task_script() -> list[dict]:
    return [
        reply(
            call("c1", "list_files", path="."),
            call("c2", "read_file", path="order.py"),
            call("c3", "read_file", path="test_order.py"),
        ),
        reply(call("c4", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("c5", "bash", command=TEST_COMMAND, cwd=".")),
        final("calculate_total now applies a percentage discount and rejects values outside 0..100."),
    ]


def test_agent_fixes_the_control_task_and_verifies_it_with_bash(tmp_path):
    agent = Agent(tmp_path, control_task_script())
    tests_before = digest(agent.file("test_order.py"))

    run = agent.start()

    assert run.status is RunStatus.COMPLETED
    assert agent.file("order.py") == FIXED_ORDER
    assert digest(agent.file("test_order.py")) == tests_before
    bash_results = [result for name, result in agent.reporter.results if name == "bash"]
    assert bash_results[-1]["command"] == TEST_COMMAND
    assert bash_results[-1]["exit_code"] == 0
    assert "OK" in bash_results[-1]["stderr"]


def test_every_tool_call_is_answered_by_a_tool_message_with_its_id(tmp_path):
    agent = Agent(tmp_path, control_task_script())
    agent.start()
    messages = agent.store.messages(agent.run_id)
    asked = [tool_call["id"] for message in messages for tool_call in message.get("tool_calls", ())]
    answered = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    assert asked == answered == ["c1", "c2", "c3", "c4", "c5"]


def test_tool_errors_reach_the_model_and_the_loop_goes_on(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(
                call("c1", "read_file", path="missing.py"),
                call("c2", "teleport", path="x"),
                call("c3", "read_file"),
                {"id": "c4", "name": "read_file", "arguments": "{not json"},
                call("c5", "bash", command="cat order.py | sh", cwd="."),
            ),
            reply(call("c6", "read_file", path="order.py")),
            final(),
        ],
    )
    run = agent.start()
    assert run.status is RunStatus.COMPLETED
    errors = [message["content"] for message in agent.tool_messages()[:5]]
    assert "no such file" in errors[0]
    assert "unknown tool: teleport" in errors[1]
    assert "$.path is required" in errors[2]
    assert "not valid JSON" in errors[3]
    assert "shell syntax is not allowed" in errors[4]
    assert '"ok": true' in agent.tool_messages()[5]["content"]


def test_execution_offers_workspace_tools_without_plan_controls(tmp_path):
    agent = Agent(tmp_path, [final()])
    agent.start()
    assert agent.model.offered == [["list_files", "read_file", "write_file", "delete_file", "bash"]]
