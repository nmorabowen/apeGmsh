"""Job handle — poll, tail, cancel, and fetch a submitted SLURM job.

A :class:`Job` survives the Python session: ``submit()`` writes a JSON
sidecar (``.apegmsh_job.json``) into the local job directory, and
``Job.load(local_dir)`` reconstructs the handle tomorrow from that file
plus the named cluster's config.

Status resolution is two-stage by necessity: ``squeue`` only knows about
live jobs, and on clusters without working accounting (``sacct``) a
finished job simply vanishes from the queue. The generated batch script
therefore writes a ``.exit_code`` sentinel in the remote job directory as
its last act; ``status()`` falls back to it once squeue comes up empty.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from . import _ssh

if TYPE_CHECKING:
    from ._cluster import Cluster

SIDECAR_NAME = ".apegmsh_job.json"
EXIT_CODE_SENTINEL = ".exit_code"


class JobStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    #: Not in the queue and no exit sentinel — crashed before the script's
    #: final write, was scancel'd mid-run, or the remote dir is gone.
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


#: squeue state -> JobStatus. Anything unlisted maps to RUNNING if the job
#: is still in the queue (it exists and is being worked on / wound down).
_SQUEUE_STATES = {
    "PENDING": JobStatus.PENDING,
    "CONFIGURING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "CANCELLED": JobStatus.CANCELLED,
    "FAILED": JobStatus.FAILED,
    "COMPLETED": JobStatus.COMPLETED,
}


@dataclass
class Job:
    """Handle to one submitted job. Construct via ``Cluster.submit`` or ``Job.load``."""

    cluster: Cluster
    name: str
    slurm_id: int
    remote_dir: str
    local_dir: Path
    deck: str

    # ------------------------------------------------------------------ io
    def save(self) -> Path:
        """Write the JSON sidecar into the local job directory."""
        sidecar = self.local_dir / SIDECAR_NAME
        payload = {
            "cluster": self.cluster.config.name,
            "config_path": self.cluster.config.config_path,
            "name": self.name,
            "slurm_id": self.slurm_id,
            "remote_dir": self.remote_dir,
            "deck": self.deck,
        }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return sidecar

    @classmethod
    def load(cls, local_dir: str | Path) -> Job:
        """Rehydrate a job handle from the sidecar written at submit time."""
        from ._cluster import Cluster

        local_dir = Path(local_dir)
        sidecar = local_dir / SIDECAR_NAME
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"no {SIDECAR_NAME} in {local_dir} — was this directory submitted "
                "with Cluster.submit()?"
            )
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        cluster = Cluster.load(
            payload["cluster"], path=payload.get("config_path") or None
        )
        return cls(
            cluster=cluster,
            name=payload["name"],
            slurm_id=int(payload["slurm_id"]),
            remote_dir=payload["remote_dir"],
            local_dir=local_dir,
            deck=payload["deck"],
        )

    # ------------------------------------------------------------ lifecycle
    def status(self) -> JobStatus:
        """Current job state: squeue while queued/running, exit sentinel after."""
        cfg = self.cluster.config
        proc = _ssh.ssh(
            cfg.ssh_host,
            f"{cfg.slurm('squeue')} -h -j {self.slurm_id} -o %T",
            check=False,  # squeue exits 1 for an unknown (finished) job id
        )
        state = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
        if state:
            return _SQUEUE_STATES.get(state, JobStatus.RUNNING)
        sentinel = PurePosixPath(self.remote_dir) / EXIT_CODE_SENTINEL
        proc = _ssh.ssh(cfg.ssh_host, f"cat {sentinel}", check=False)
        code = proc.stdout.strip()
        if proc.returncode != 0 or not code:
            return JobStatus.UNKNOWN
        return JobStatus.COMPLETED if code == "0" else JobStatus.FAILED

    def wait(
        self, *, poll: float = 15.0, timeout: float | None = None
    ) -> JobStatus:
        """Block until the job reaches a terminal state; return it.

        Polls :meth:`status` every ``poll`` seconds. With ``timeout``
        (seconds), raises :class:`TimeoutError` if the job is still
        non-terminal when it elapses — the job itself keeps running;
        use :meth:`cancel` if you also want it gone.
        """
        import time

        deadline = time.monotonic() + timeout if timeout is not None else None
        status = self.status()
        while not status.is_terminal:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {self.name!r} (slurm {self.slurm_id}) still "
                    f"{status.value} after {timeout}s; it keeps running on "
                    "the cluster — poll later with Job.load(...).status()"
                )
            time.sleep(poll)
            status = self.status()
        return status

    def tail(self, n: int = 50, *, stream: str = "out") -> str:
        """Last ``n`` lines of the remote stdout (``stream="out"``) or stderr."""
        if stream not in ("out", "err"):
            raise ValueError(f"stream must be 'out' or 'err', got {stream!r}")
        log = PurePosixPath(self.remote_dir) / f"{self.name}-{self.slurm_id}.{stream}"
        proc = _ssh.ssh(self.cluster.config.ssh_host, f"tail -n {int(n)} {log}", check=False)
        if proc.returncode != 0:
            return f"<no {stream} log yet: {proc.stderr.strip()}>"
        return proc.stdout

    def cancel(self) -> None:
        """scancel the job."""
        cfg = self.cluster.config
        _ssh.ssh(cfg.ssh_host, f"{cfg.slurm('scancel')} {self.slurm_id}")

    def fetch(self, dest: str | Path | None = None) -> Path:
        """Pull the whole remote job directory back. Defaults into ``local_dir``."""
        dest_path = Path(dest) if dest is not None else self.local_dir
        _ssh.pull_dir(self.cluster.config.ssh_host, self.remote_dir, dest_path)
        return dest_path

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Job(name={self.name!r}, slurm_id={self.slurm_id}, "
            f"cluster={self.cluster.config.name!r})"
        )
