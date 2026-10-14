import os
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from .budget import Budget
from .files import write_atomically
from .policy import ConfigError, Policy

_SOURCE_PROBES = ("module.py", "test_module.py")


@dataclass(frozen=True)
class Limits:
    bash_timeout_seconds: float
    tool_output_bytes: int


@dataclass(frozen=True)
class Settings:
    """Everything a run needs from the policy, resolved against one workspace.

    `policy_text` is the policy as it was when the run started; a run keeps those rules even if the file changes.
    """

    workspace: Path
    policy_path: Path
    policy_text: str
    policy: Policy
    limits: Limits
    budget: Budget

    def read_policy_file(self) -> bytes | None:
        try:
            return self.policy_path.read_bytes()
        except OSError:
            return None

    def restore_policy_file(self, before: bytes | None) -> str | None:
        """Undo an edit an action made to the policy file and describe it, or return None if the file is as it was."""
        if before is None or self.read_policy_file() == before:
            return None
        try:
            write_atomically(self.policy_path, before)
        except OSError as error:
            return f"the action modified the policy file and restoring it failed: {error}"
        return "the action modified the policy file; the runtime restored it and stopped the run"


def load_settings(policy_path: Path, workspace: Path) -> Settings:
    try:
        text = policy_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read policy {policy_path}: {error}") from None
    return parse_settings(text, policy_path, workspace)


def parse_settings(text: str, policy_path: Path, workspace: Path) -> Settings:
    policy_path = policy_path.resolve()
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise ConfigError(f"workspace does not exist: {workspace}")
    if policy_path.is_relative_to(workspace):
        raise ConfigError("the policy file must live outside the workspace")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"cannot parse policy {policy_path}: {error}") from None
    generated = _strings(data, "journal", "ignore", default=["__pycache__", "*.pyc"])
    _reject_hidden_sources(generated, workspace)
    return Settings(
        workspace=workspace,
        policy_path=policy_path,
        policy_text=text,
        policy=Policy(
            workspace,
            allow_commands=_strings(data, "commands", "allow"),
            deny_commands=_strings(data, "commands", "deny"),
            allow_dirs=_strings(data, "directories", "allow"),
            deny_dirs=_strings(data, "directories", "deny"),
            generated=generated,
        ),
        limits=Limits(
            bash_timeout_seconds=_positive(data, "limits", "bash_timeout_seconds", (int, float)),
            tool_output_bytes=_positive(data, "limits", "tool_output_bytes", int),
        ),
        budget=Budget(**{name: _positive(data, "budget", name, int) for name in Budget.counters()}),
    )


def _reject_hidden_sources(patterns: list[str], workspace: Path) -> None:
    """Refuse journal exclusions that would cover any Python file, present in the workspace or hypothetical."""
    names = set(_SOURCE_PROBES)
    for directory, _, files in os.walk(workspace):
        relative = Path(directory).relative_to(workspace)
        for file in files:
            if file.endswith(".py"):
                names.update((relative / file).parts)
    hidden = sorted(pattern for pattern in patterns if any(fnmatchcase(name, pattern) for name in names))
    if hidden:
        raise ConfigError(f"journal.ignore must not hide Python sources or tests: {', '.join(hidden)}")


def _value(data: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = data.get(section, {}).get(key, default)
    if value is None:
        raise ConfigError(f"policy is missing {section}.{key}")
    return value


def _strings(data: dict[str, Any], section: str, key: str, default: list[str] | None = None) -> list[str]:
    value = _value(data, section, key, default)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{section}.{key} must be a list of strings")
    return value


def _positive(data: dict[str, Any], section: str, key: str, kind: type | tuple[type, ...]) -> Any:
    value = _value(data, section, key)
    if isinstance(value, bool) or not isinstance(value, kind) or value <= 0:
        raise ConfigError(f"{section}.{key} must be a positive number")
    return value
