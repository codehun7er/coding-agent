import os
import tempfile
from pathlib import Path


def write_atomically(target: Path, content: bytes) -> None:
    """Replace a file in one rename through a fresh temporary file, so a crash leaves the old or the new content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o7777 if target.exists() else 0o666 & ~_umask()
    descriptor, staging = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".partial")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.chmod(staging, mode)
        os.replace(staging, target)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current
