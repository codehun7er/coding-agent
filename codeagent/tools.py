import contextlib
import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any

from .files import write_atomically
from .policy import Action, Effect
from .schema import function_spec, object_schema
from .settings import Settings

READ_FILE = "read_file"
BASH = "bash"


class ToolFailure(Exception):
    pass


@dataclass(frozen=True)
class Outcome:
    """A tool result: fixed facts in `fields`, unbounded text in `streams`, which is what the output budget meters."""

    ok: bool
    fields: dict[str, Any] = field(default_factory=dict)
    streams: dict[str, str] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def failure(cls, error: str) -> "Outcome":
        return cls(False, {"error": error})

    @property
    def size(self) -> int:
        return sum(len(text.encode()) for text in self.streams.values())

    def clip(self, limit: int) -> "Outcome":
        if self.size <= limit:
            return self
        clipped: dict[str, str] = {}
        left = limit
        for name, text in self.streams.items():
            clipped[name] = text.encode()[:left].decode(errors="ignore")
            left -= len(clipped[name].encode())
        return replace(self, streams=clipped, truncated=True)

    def render(self) -> str:
        payload = {"ok": self.ok, **self.fields, **self.streams}
        if self.truncated:
            payload["truncated"] = "output is incomplete: it was cut to fit the output limit"
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class Tool:
    """A workspace tool. `retryable` marks tools whose repeated execution leaves the workspace in the same state."""

    name: str
    description: str
    parameters: dict[str, Any]
    effect: Effect
    retryable: bool
    metered: bool
    execute: Callable[[dict[str, Any], int], Outcome]

    def action(self, arguments: dict[str, Any]) -> Action:
        if self.effect is Effect.EXEC:
            return Action(self.effect, command=arguments["command"], cwd=arguments["cwd"])
        return Action(self.effect, paths=(_path(arguments),))

    def spec(self) -> dict[str, Any]:
        return function_spec(self.name, self.description, self.parameters)


_PATH = {"type": "string", "minLength": 1, "description": "Path relative to the workspace root"}


class Toolbox:
    """The workspace tools. Executors take an output limit in bytes and re-resolve every path through the policy."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._policy = settings.policy
        tools = (
            Tool(
                name="list_files",
                description="List files under a workspace directory recursively.",
                parameters=object_schema({"path": _PATH}, ()),
                effect=Effect.READ,
                retryable=True,
                metered=True,
                execute=self._list_files,
            ),
            Tool(
                name=READ_FILE,
                description="Read a UTF-8 text file from the workspace.",
                parameters=object_schema({"path": _PATH}, ("path",)),
                effect=Effect.READ,
                retryable=True,
                metered=True,
                execute=self._read_file,
            ),
            Tool(
                name="write_file",
                description="Create a file or replace its whole content.",
                parameters=object_schema({"path": _PATH, "content": {"type": "string"}}, ("path", "content")),
                effect=Effect.WRITE,
                retryable=True,
                metered=False,
                execute=self._write_file,
            ),
            Tool(
                name="delete_file",
                description="Delete a file from the workspace.",
                parameters=object_schema({"path": _PATH}, ("path",)),
                effect=Effect.DELETE,
                retryable=True,
                metered=False,
                execute=self._delete_file,
            ),
            Tool(
                name=BASH,
                description=(
                    "Run one command from the allow list through Bash and return stdout, stderr and the exit code. "
                    "Shell syntax such as pipes, redirects and substitutions is rejected."
                ),
                parameters=object_schema(
                    {
                        "command": {"type": "string", "minLength": 1},
                        "cwd": {"type": "string", "minLength": 1, "description": "Working directory, '.' is the root"},
                    },
                    ("command", "cwd"),
                ),
                effect=Effect.EXEC,
                retryable=False,
                metered=True,
                execute=self._bash,
            ),
        )
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self, effects: Collection[Effect]) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values() if tool.effect in effects]

    def _list_files(self, arguments: dict[str, Any], limit: int) -> Outcome:
        base = self._policy.locate(_path(arguments))
        if not base.is_dir():
            raise ToolFailure(f"not a directory: {_path(arguments)}")
        entries = "\n".join(sorted(self._walk(base)))
        return Outcome(True, {"path": self._policy.display(base)}, {"entries": entries}).clip(limit)

    def _walk(self, directory: Path) -> list[str]:
        entries: list[str] = []
        for child in directory.iterdir():
            relative = self._policy.display(child)
            if child.is_symlink() or self._policy.is_generated(child) or not self._policy.permits(relative):
                continue
            if child.is_dir():
                entries.append(relative + "/")
                entries.extend(self._walk(child))
            else:
                entries.append(relative)
        return entries

    def _read_file(self, arguments: dict[str, Any], limit: int) -> Outcome:
        path = self._policy.locate(arguments["path"])
        if not path.is_file():
            raise ToolFailure(f"no such file: {arguments['path']}")
        with path.open("rb") as stream:
            content = _head(stream, limit)
        return Outcome(True, {"path": self._policy.display(path)}, {"content": content}).clip(limit)

    def _write_file(self, arguments: dict[str, Any], limit: int) -> Outcome:
        path = self._policy.locate(arguments["path"])
        if path.is_dir():
            raise ToolFailure(f"is a directory: {arguments['path']}")
        created = not path.exists()
        data = arguments["content"].encode()
        write_atomically(path, data)
        return Outcome(True, {"path": self._policy.display(path), "bytes": len(data), "created": created})

    def _delete_file(self, arguments: dict[str, Any], limit: int) -> Outcome:
        path = self._policy.locate(arguments["path"])
        if not path.is_file():
            raise ToolFailure(f"no such file: {arguments['path']}")
        path.unlink()
        return Outcome(True, {"path": self._policy.display(path), "deleted": True})

    def _bash(self, arguments: dict[str, Any], limit: int) -> Outcome:
        """Run the command with output going to temporary files, so neither a timeout nor a stray child can hang it."""
        command, cwd = self._policy.command(arguments["command"], arguments["cwd"])
        timeout = self._settings.limits.bash_timeout_seconds
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self._settings.workspace),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        fields: dict[str, Any] = {"command": command, "cwd": self._policy.display(cwd)}
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(
                    ["bash", "-c", command],
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except FileNotFoundError:
                raise ToolFailure("bash is not available") from None
            try:
                fields["exit_code"] = process.wait(timeout=timeout)
                ok = True
            except subprocess.TimeoutExpired:
                fields["exit_code"] = None
                fields["error"] = f"timed out after {timeout} seconds"
                ok = False
            _kill_group(process.pid)
            process.wait()
            stdout.seek(0)
            stderr.seek(0)
            streams = {"stdout": _head(stdout, limit), "stderr": _head(stderr, limit)}
        return Outcome(ok, fields, streams).clip(limit)


def _path(arguments: dict[str, Any]) -> str:
    return arguments.get("path", ".")


def _head(stream: IO[bytes], limit: int) -> str:
    return stream.read(limit + 1).decode(errors="replace")


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
