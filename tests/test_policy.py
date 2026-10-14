import os
from pathlib import Path

import pytest

from codeagent.policy import Action, ConfigError, Decision, Effect
from codeagent.records import Mode
from codeagent.settings import load_settings

from .support import ROOT, TEST_COMMAND, make_project, policy_text


@pytest.fixture
def settings(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path)
    (workspace / "private").mkdir()
    (workspace / "private" / "notes.txt").write_text("hidden")
    (workspace / "pkg").mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    return load_settings(policy_path, workspace)


def check(settings, action: Action, mode: Mode = Mode.AUTO, planning: bool = False):
    return settings.policy.check(action, mode, planning=planning)


def bash(command: str, cwd: str = ".") -> Action:
    return Action(Effect.EXEC, command=command, cwd=cwd)


def test_allowed_command_passes(settings):
    assert check(settings, bash(TEST_COMMAND)).decision is Decision.ALLOW


def test_extra_whitespace_does_not_change_the_command(settings):
    assert check(settings, bash("python3  -m unittest   -v")).decision is Decision.ALLOW


def test_command_outside_the_allow_list_is_denied(settings):
    verdict = check(settings, bash("python3 order.py"))
    assert verdict.decision is Decision.DENY
    assert "not in the allow list" in verdict.reason


def test_deny_rule_wins_over_allow_rule(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path, allow_commands=("rm order.py",), deny_commands=("rm",))
    verdict = load_settings(policy_path, workspace).policy.check(bash("rm order.py"), Mode.AUTO, planning=False)
    assert verdict.decision is Decision.DENY
    assert "deny rule 'rm'" in verdict.reason


def test_deny_rule_matches_nested_commands(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path, allow_commands=("env rm order.py",), deny_commands=("rm",))
    verdict = load_settings(policy_path, workspace).policy.check(bash("env rm order.py"), Mode.AUTO, planning=False)
    assert verdict.decision is Decision.DENY


@pytest.mark.parametrize(
    "command",
    [
        f"{TEST_COMMAND} | tee log",
        f"{TEST_COMMAND}; rm order.py",
        f"{TEST_COMMAND} && rm order.py",
        f"{TEST_COMMAND} > out.txt",
        "echo $(rm order.py)",
        "echo `id`",
        "bash -c 'rm order.py'",
        f"{TEST_COMMAND}\nrm order.py",
    ],
)
def test_shell_syntax_is_rejected(settings, command: str):
    assert check(settings, bash(command)).decision is Decision.DENY


def test_allowed_command_in_a_denied_directory_is_denied(settings):
    verdict = check(settings, bash("ls", cwd="private"))
    assert verdict.decision is Decision.DENY
    assert "denied by policy: private" in verdict.reason


def test_command_path_arguments_are_checked(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path, allow_commands=("ls private", "ls ../", "ls --dir=/etc"))
    policy = load_settings(policy_path, workspace).policy
    (workspace / "private").mkdir()
    for command in ("ls private", "ls ../", "ls --dir=/etc"):
        assert policy.check(bash(command), Mode.AUTO, planning=False).decision is Decision.DENY


def test_file_tool_cannot_enter_a_denied_subdirectory(settings):
    verdict = check(settings, Action(Effect.READ, paths=("private/notes.txt",)))
    assert verdict.decision is Decision.DENY
    assert "private" in verdict.reason


@pytest.mark.parametrize("path", ["../outside.txt", "pkg/../../outside.txt", "pkg/../order.py"])
def test_parent_references_are_denied(settings, path: str):
    verdict = check(settings, Action(Effect.WRITE, paths=(path,)))
    assert verdict.decision is Decision.DENY
    assert "'..'" in verdict.reason


def test_absolute_paths_are_denied(settings, tmp_path: Path):
    for path in (str(tmp_path / "outside.txt"), str(settings.policy_path), "~/x"):
        assert check(settings, Action(Effect.WRITE, paths=(path,))).decision is Decision.DENY


def test_symlink_escaping_the_workspace_is_denied(settings, tmp_path: Path):
    os.symlink(tmp_path / "outside.txt", settings.workspace / "link.txt")
    verdict = check(settings, Action(Effect.READ, paths=("link.txt",)))
    assert verdict.decision is Decision.DENY
    assert "symlink" in verdict.reason


def test_symlinked_directory_is_denied_even_inside_the_workspace(settings):
    os.symlink(settings.workspace / "private", settings.workspace / "alias")
    assert check(settings, Action(Effect.READ, paths=("alias/notes.txt",))).decision is Decision.DENY


@pytest.mark.parametrize(
    ("mode", "action", "decision"),
    [
        (Mode.SAFE, Action(Effect.READ, paths=("order.py",)), Decision.ALLOW),
        (Mode.SAFE, Action(Effect.WRITE, paths=("order.py",)), Decision.CONFIRM),
        (Mode.SAFE, bash(TEST_COMMAND), Decision.CONFIRM),
        (Mode.SAFE, Action(Effect.DELETE, paths=("order.py",)), Decision.CONFIRM),
        (Mode.AUTO, Action(Effect.READ, paths=("order.py",)), Decision.ALLOW),
        (Mode.AUTO, Action(Effect.WRITE, paths=("order.py",)), Decision.ALLOW),
        (Mode.AUTO, bash(TEST_COMMAND), Decision.ALLOW),
        (Mode.AUTO, Action(Effect.DELETE, paths=("order.py",)), Decision.CONFIRM),
    ],
)
def test_mode_decides_what_needs_confirmation(settings, mode: Mode, action: Action, decision: Decision):
    assert check(settings, action, mode).decision is decision


def test_planning_denies_every_side_effect(settings):
    for action in (Action(Effect.WRITE, paths=("order.py",)), bash(TEST_COMMAND)):
        verdict = check(settings, action, planning=True)
        assert verdict.decision is Decision.DENY
        assert "plan mode" in verdict.reason
    assert check(settings, Action(Effect.READ, paths=("order.py",)), planning=True).decision is Decision.ALLOW


def test_allowed_directories_must_be_inside_the_workspace(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path, allow_dirs=("..",))
    with pytest.raises(ConfigError, match="inside the workspace"):
        load_settings(policy_path, workspace)


def test_policy_file_must_live_outside_the_workspace(tmp_path: Path):
    workspace, _ = make_project(tmp_path)
    inside = workspace / "policy.toml"
    inside.write_text(policy_text())
    with pytest.raises(ConfigError, match="outside the workspace"):
        load_settings(inside, workspace)


def test_allow_rules_cannot_contain_shell_syntax(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path, allow_commands=("ls | sh",))
    with pytest.raises(ConfigError, match="invalid command rule"):
        load_settings(policy_path, workspace)


def test_journal_ignore_cannot_hide_sources(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path)
    policy_path.write_text(policy_text().replace('ignore = ["__pycache__", "*.pyc"]', 'ignore = ["*.py"]'))
    with pytest.raises(ConfigError, match="must not hide"):
        load_settings(policy_path, workspace)


def test_repository_policy_example_loads():
    settings = load_settings(ROOT / "policy.toml", ROOT / "workspace")
    assert TEST_COMMAND in settings.policy.allowed_commands


@pytest.mark.parametrize("path", ["PRIVATE/notes.txt", "Private/notes.txt", "pRiVaTe"])
def test_denied_directory_matches_regardless_of_letter_case(settings, path: str):
    assert check(settings, Action(Effect.READ, paths=(path,))).decision is Decision.DENY


def test_tools_cannot_change_files_excluded_from_the_journal(settings):
    for path in ("__pycache__/order.cpython-313.pyc", "evil.pyc"):
        verdict = check(settings, Action(Effect.WRITE, paths=(path,)))
        assert verdict.decision is Decision.DENY
        assert "outside the change journal" in verdict.reason
    assert check(settings, Action(Effect.READ, paths=("evil.pyc",))).decision is Decision.ALLOW


def test_journal_ignore_cannot_hide_files_present_in_the_workspace(tmp_path: Path):
    workspace, policy_path = make_project(tmp_path)
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("")
    for pattern in ("order.py", "src"):
        policy_path.write_text(policy_text().replace('ignore = ["__pycache__", "*.pyc"]', f'ignore = ["{pattern}"]'))
        with pytest.raises(ConfigError, match="must not hide"):
            load_settings(policy_path, workspace)


def test_generated_names_match_regardless_of_letter_case(settings):
    verdict = check(settings, Action(Effect.WRITE, paths=("__PYCACHE__/evil.py",)))
    assert verdict.decision is Decision.DENY
