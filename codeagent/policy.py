import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from .records import Mode


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXEC = "exec"

    @property
    def has_side_effect(self) -> bool:
        return self is not Effect.READ


class Decision(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""


@dataclass(frozen=True)
class Action:
    """What a tool call would do, reduced to the facts that policy rules are written about."""

    effect: Effect
    paths: tuple[str, ...] = ()
    command: str | None = None
    cwd: str = "."


class Violation(Exception):
    """A path or command the policy refuses; the message is the reason the model receives."""


class ConfigError(ValueError):
    pass


_PLAIN_WORD = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")

_GATES = {
    Mode.SAFE: {
        Effect.READ: Decision.ALLOW,
        Effect.WRITE: Decision.CONFIRM,
        Effect.EXEC: Decision.CONFIRM,
        Effect.DELETE: Decision.CONFIRM,
    },
    Mode.AUTO: {
        Effect.READ: Decision.ALLOW,
        Effect.WRITE: Decision.ALLOW,
        Effect.EXEC: Decision.ALLOW,
        Effect.DELETE: Decision.CONFIRM,
    },
}


def command_words(command: str) -> tuple[str, ...]:
    """Split a command into plain words, refusing anything Bash would treat as syntax."""
    words = tuple(command.split())
    if not words:
        raise Violation("command is empty")
    for word in words:
        if not _PLAIN_WORD.fullmatch(word):
            raise Violation(f"shell syntax is not allowed in {word!r}: only exact commands from the allow list run")
    return words


class Policy:
    """Rules for commands, directories and generated files of one workspace.

    Deny beats allow and anything unlisted is denied. Denied directories match regardless of letter case and
    Unicode normalization, so a case-insensitive file system cannot open a second way into them.
    """

    def __init__(
        self,
        root: Path,
        *,
        allow_commands: Iterable[str],
        deny_commands: Iterable[str],
        allow_dirs: Iterable[str],
        deny_dirs: Iterable[str],
        generated: Iterable[str],
    ) -> None:
        self.root = root.resolve()
        self._allow_commands = frozenset(self._rule(command) for command in allow_commands)
        self._deny_commands = tuple(self._rule(command) for command in deny_commands)
        self._allow_dirs = tuple(self._directory(raw, inside_root=True) for raw in allow_dirs)
        self._deny_dirs = tuple(self._directory(raw, inside_root=False) for raw in deny_dirs)
        self._generated = tuple(generated)

    @property
    def allowed_commands(self) -> tuple[str, ...]:
        return tuple(sorted(" ".join(words) for words in self._allow_commands))

    @staticmethod
    def allowed_effects(*, planning: bool) -> frozenset[Effect]:
        return frozenset({Effect.READ}) if planning else frozenset(Effect)

    def check(self, action: Action, mode: Mode, *, planning: bool) -> Verdict:
        if action.effect not in self.allowed_effects(planning=planning):
            return Verdict(Decision.DENY, "plan mode is read-only until the user approves the plan")
        try:
            for path in action.paths:
                target = self.locate(path)
                if action.effect in (Effect.WRITE, Effect.DELETE) and self.is_generated(target):
                    raise Violation(f"generated files are outside the change journal and cannot be changed: {path}")
            if action.command is not None:
                self.command(action.command, action.cwd)
        except Violation as violation:
            return Verdict(Decision.DENY, str(violation))
        return Verdict(_GATES[mode][action.effect])

    def confine(self, raw: str, base: str = ".") -> Path:
        """Resolve a workspace-relative path, refusing absolute paths, '..', symlinks and anything outside the root."""
        if not raw or "\0" in raw:
            raise Violation("path is empty or malformed")
        if raw.startswith(("/", "~")):
            raise Violation(f"absolute paths are not allowed: {raw}")
        relative = PurePosixPath(base, raw)
        if ".." in relative.parts:
            raise Violation(f"'..' is not allowed in paths: {raw}")
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise Violation(f"symlinks are not allowed: {raw}")
        canonical = current.resolve()
        if not canonical.is_relative_to(self.root):
            raise Violation(f"path escapes the workspace: {raw}")
        return canonical

    def locate(self, raw: str, base: str = ".") -> Path:
        """Resolve a path like `confine` and additionally enforce the allowed and denied directories."""
        canonical = self.confine(raw, base)
        folded = _folded(canonical)
        denied = next((directory for directory in self._deny_dirs if folded.is_relative_to(_folded(directory))), None)
        if denied is not None:
            raise Violation(f"directory is denied by policy: {self.display(denied)}")
        if not any(canonical.is_relative_to(directory) for directory in self._allow_dirs):
            raise Violation(f"path is outside the allowed directories: {raw}")
        return canonical

    def permits(self, raw: str) -> bool:
        try:
            self.locate(raw)
        except Violation:
            return False
        return True

    def audit(self, changed: Iterable[str]) -> str | None:
        """Name the files an action changed that the directory rules forbid, which only a command can reach."""
        forbidden = [path for path in changed if not self.permits(path)]
        if not forbidden:
            return None
        return f"the action changed files the policy does not allow: {', '.join(forbidden)}; roll the run back"

    def is_generated(self, path: Path) -> bool:
        relative = path.relative_to(self.root) if path.is_absolute() else path
        return any(fnmatchcase(_fold(part), _fold(pattern)) for part in relative.parts for pattern in self._generated)

    def command(self, command: str, cwd: str) -> tuple[str, Path]:
        """Validate a command in a working directory and return its normalized text and canonical directory."""
        directory = self.locate(cwd)
        if not directory.is_dir():
            raise Violation(f"working directory does not exist: {cwd}")
        words = command_words(command)
        for rule in self._deny_commands:
            if _contains(words, rule):
                raise Violation(f"command matches deny rule {' '.join(rule)!r}")
        if words not in self._allow_commands:
            raise Violation(f"command is not in the allow list: {' '.join(words)!r}")
        for word in words[1:]:
            for fragment in (word, word.partition("=")[2]):
                if fragment and _names_path(fragment, directory):
                    self.locate(fragment, base=cwd)
        return " ".join(words), directory

    def display(self, path: Path) -> str:
        if path.is_relative_to(self.root):
            return path.relative_to(self.root).as_posix() or "."
        return str(path)

    def _rule(self, command: str) -> tuple[str, ...]:
        try:
            return command_words(command)
        except Violation as violation:
            raise ConfigError(f"invalid command rule {command!r}: {violation}") from None

    def _directory(self, raw: str, *, inside_root: bool) -> Path:
        directory = (self.root / raw).resolve()
        if inside_root and not directory.is_relative_to(self.root):
            raise ConfigError(f"allowed directory must be inside the workspace: {raw}")
        return directory


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _folded(path: Path) -> PurePosixPath:
    return PurePosixPath(_fold(path.as_posix()))


def _contains(words: tuple[str, ...], rule: tuple[str, ...]) -> bool:
    return any(words[start : start + len(rule)] == rule for start in range(len(words) - len(rule) + 1))


def _names_path(word: str, directory: Path) -> bool:
    if word.startswith(("/", "~", ".")) or "/" in word:
        return True
    candidate = directory / word
    return candidate.is_symlink() or candidate.exists()
