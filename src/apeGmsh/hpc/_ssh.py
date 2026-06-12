"""Subprocess transport over the native ``ssh`` / ``scp`` / ``tar`` binaries.

Every external command in :mod:`apeGmsh.hpc` funnels through
:func:`run_local` — one seam for tests to monkeypatch, and one place where
the no-interactive-prompt policy (``BatchMode=yes``) is enforced. Using the
OS executables (Windows ships OpenSSH and bsdtar) means the user's
``~/.ssh/config`` aliases and key agent work untouched, with zero Python
dependencies.

Directory transfer goes as a single gzipped tarball (local tar -> scp ->
remote untar) instead of ``scp -r``: one round-trip regardless of file
count, deterministic target naming, and no rsync requirement on Windows.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path, PurePosixPath


class HPCError(RuntimeError):
    """A remote (or transfer) command failed."""


def run_local(
    argv: list[str],
    *,
    check: bool = True,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run a local command (ssh/scp/tar). The single subprocess seam."""
    proc = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,  # ssh forwards inherited stdin; never let it block
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise HPCError(
            f"command failed (exit {proc.returncode}): {' '.join(argv)}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    return proc


def ssh(
    host: str,
    command: str,
    *,
    check: bool = True,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` on ``host`` non-interactively.

    ``BatchMode=yes`` makes a missing/declined key a fast failure instead of
    a hung password prompt.
    """
    return run_local(
        ["ssh", "-o", "BatchMode=yes", host, command],
        check=check,
        timeout=timeout,
    )


def push_dir(host: str, local_dir: Path, remote_dir: str) -> None:
    """Copy the *contents* of ``local_dir`` into ``remote_dir`` (created)."""
    remote_tar = str(PurePosixPath(remote_dir) / ".apegmsh_push.tgz")
    with tempfile.TemporaryDirectory() as tmp:
        local_tar = Path(tmp) / "push.tgz"
        run_local(["tar", "-czf", str(local_tar), "-C", str(local_dir), "."])
        ssh(host, f"mkdir -p {remote_dir}")
        run_local(["scp", "-q", "-o", "BatchMode=yes", str(local_tar), f"{host}:{remote_tar}"])
    ssh(host, f"tar -xzf {remote_tar} -C {remote_dir} && rm {remote_tar}")


def pull_dir(host: str, remote_dir: str, local_dir: Path) -> None:
    """Copy the *contents* of ``remote_dir`` into ``local_dir`` (created)."""
    # The tarball must live OUTSIDE the directory being tarred — GNU tar
    # exits 1 ("file changed as we read it") when its own output mutates
    # the tree it is reading.
    remote_tar = f"/tmp/apegmsh_pull_{PurePosixPath(remote_dir).name}.tgz"
    local_dir.mkdir(parents=True, exist_ok=True)
    ssh(host, f"tar -czf {remote_tar} -C {remote_dir} .")
    with tempfile.TemporaryDirectory() as tmp:
        local_tar = Path(tmp) / "pull.tgz"
        run_local(["scp", "-q", "-o", "BatchMode=yes", f"{host}:{remote_tar}", str(local_tar)])
        ssh(host, f"rm -f {remote_tar}")
        run_local(["tar", "-xzf", str(local_tar), "-C", str(local_dir)])
