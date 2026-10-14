import difflib
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .files import write_atomically
from .records import FileChange
from .settings import Settings
from .store import Store


class Conflict(Exception):
    """Files differ from the state the journal last recorded, so acting on them could overwrite someone else's edit."""

    def __init__(self, message: str, paths: Sequence[str]) -> None:
        super().__init__(message)
        self.paths = tuple(paths)


@dataclass(frozen=True)
class Diff:
    paths: tuple[str, ...]
    text: str


class Journal:
    """Records what a run does to workspace files and undoes it on request unless someone else edited them since."""

    def __init__(self, store: Store, settings: Settings) -> None:
        self._store = store
        self._policy = settings.policy
        self._root = settings.workspace

    def begin(self, run_id: str) -> None:
        with self._store.atomic():
            self._store.update_run(run_id, baseline=self._capture(run_id))

    def guard(self, run_id: str) -> int:
        """Snapshot the workspace before a side effect, raising Conflict if it drifted from the last recorded state."""
        drift = self.drift(run_id)
        if drift.paths:
            raise Conflict(f"files changed outside the agent: {', '.join(drift.paths)}", drift.paths)
        return self._capture(run_id)

    def record(self, run_id: str, pre: int) -> Diff:
        """Attribute everything that changed since snapshot `pre` to the run and make the result the new baseline."""
        with self._store.atomic():
            post = self._capture(run_id)
            before, after = self._store.snapshot(pre), self._store.snapshot(post)
            diff = self._compare(before, after)
            for path in diff.paths:
                self._store.record_change(run_id, path, before.get(path), after.get(path))
            self._store.update_run(run_id, baseline=post)
            return diff

    def drift(self, run_id: str) -> Diff:
        """Show how the workspace differs from the run's last recorded state without recording anything."""
        baseline = self._store.run(run_id).baseline
        return self._compare(self._store.snapshot(baseline), self._scan())

    def settle(self, run_id: str, *, adopt: bool) -> Diff:
        """Accept drift into the baseline: attributed to the run when `adopt`, treated as external otherwise."""
        baseline = self._store.run(run_id).baseline
        if adopt:
            return self.record(run_id, baseline)
        with self._store.atomic():
            post = self._capture(run_id)
            self._store.update_run(run_id, baseline=post)
            return self._compare(self._store.snapshot(baseline), self._store.snapshot(post))

    def diff(self, run_id: str) -> str:
        return "".join(
            _unified(change.path, self._content(change.original), self._content(change.final))
            for change in self._net_changes(run_id)
        )

    def rollback(self, run_id: str) -> tuple[str, ...]:
        """Put every file the run changed back to its original state, or write nothing if any was edited since.

        A file already at its original state counts as restored, so a rollback cut short by a crash can be rerun.
        Deletions go first, so on a case-insensitive file system a restored name cannot land in a file about to go.
        Paths are re-confined to the workspace; the allow and deny lists do not apply, because undoing the run's
        own change cannot reach anything the run did not already touch.
        """
        changes = self._net_changes(run_id)
        targets = {change.path: self._policy.confine(change.path) for change in changes}
        current = self._scan()
        conflicts = [
            change.path for change in changes if current.get(change.path) not in (change.final, change.original)
        ]
        if conflicts:
            raise Conflict(f"files changed after the run's last action: {', '.join(conflicts)}", conflicts)
        pending = [change for change in changes if current.get(change.path) != change.original]
        for change in pending:
            if change.original is None:
                targets[change.path].unlink()
        for change in pending:
            if change.original is not None:
                write_atomically(targets[change.path], self._store.blob(change.original))
        with self._store.atomic():
            self._store.update_run(run_id, baseline=self._capture(run_id))
        return tuple(targets)

    def _net_changes(self, run_id: str) -> list[FileChange]:
        return [change for change in self._store.file_changes(run_id) if change.original != change.final]

    def _capture(self, run_id: str) -> int:
        return self._store.save_snapshot(run_id, self._scan())

    def _scan(self) -> dict[str, str]:
        """Hash every tracked file and store its content, so any recorded state can be diffed or restored later."""
        manifest: dict[str, str] = {}
        with self._store.atomic():
            for path in self._files():
                content = (self._root / path).read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                self._store.put_blob(digest, content)
                manifest[path] = digest
        return manifest

    def _files(self) -> list[str]:
        files: list[str] = []
        for directory, subdirectories, names in os.walk(self._root):
            base = Path(directory)
            subdirectories[:] = sorted(name for name in subdirectories if self._tracks(base / name))
            tracked = sorted(name for name in names if self._tracks(base / name))
            files.extend((base / name).relative_to(self._root).as_posix() for name in tracked)
        return files

    def _tracks(self, path: Path) -> bool:
        return not path.is_symlink() and not self._policy.is_generated(path)

    def _compare(self, before: Mapping[str, str], after: Mapping[str, str]) -> Diff:
        paths = tuple(sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path)))
        text = "".join(
            _unified(path, self._content(before.get(path)), self._content(after.get(path))) for path in paths
        )
        return Diff(paths, text)

    def _content(self, digest: str | None) -> bytes | None:
        return None if digest is None else self._store.blob(digest)


def _unified(path: str, before: bytes | None, after: bytes | None) -> str:
    if b"\0" in (before or b"") or b"\0" in (after or b""):
        return f"Binary file {path} changed\n"
    lines = difflib.unified_diff(
        _lines(before),
        _lines(after),
        fromfile="/dev/null" if before is None else f"a/{path}",
        tofile="/dev/null" if after is None else f"b/{path}",
        lineterm="",
    )
    return "".join(line + "\n" for line in lines)


def _lines(content: bytes | None) -> list[str]:
    return [] if content is None else content.decode(errors="replace").splitlines()
