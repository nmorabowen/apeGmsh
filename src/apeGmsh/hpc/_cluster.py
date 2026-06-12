"""Cluster facade — render the batch script, push the job dir, sbatch it.

The generated ``job.sbatch`` is written into the *local* job directory
before the push, so what ran is always inspectable (and re-runnable by
hand) next to the deck. It is written with LF line endings and UTF-8
explicitly: this module's primary platform is Windows-authored scripts
executed by remote bash, where a stray CR breaks the shebang line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import _ssh
from ._config import ClusterConfig
from ._job import Job
from ._ssh import HPCError

#: Job names land in filesystem paths, sbatch directives, and shell command
#: strings — keep them boring.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")


@dataclass(frozen=True)
class Cluster:
    """One remote SLURM cluster. ``Cluster.load("esmeralda")`` to construct."""

    config: ClusterConfig

    @classmethod
    def load(cls, name: str, path: str | Path | None = None) -> Cluster:
        return cls(ClusterConfig.load(name, path=path))

    # ----------------------------------------------------------------- ssh
    def ping(self) -> bool:
        """True if the host answers a trivial command over BatchMode ssh."""
        proc = _ssh.ssh(self.config.ssh_host, "echo ok", check=False, timeout=30.0)
        return proc.returncode == 0 and proc.stdout.strip() == "ok"

    # -------------------------------------------------------------- render
    def render_batch_script(
        self,
        *,
        name: str,
        np: int,
        deck: str,
        binary: str | None = None,
        walltime: str | None = None,
    ) -> str:
        """The sbatch script for one job, as a string (LF newlines).

        The trailing ``.exit_code`` write is load-bearing: on a cluster
        without working ``sacct`` it is the only durable record of how the
        job ended (see ``Job.status``).
        """
        cfg = self.config
        directives = [
            f"#SBATCH --job-name={name}",
            f"#SBATCH --ntasks={np}",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --output=%x-%j.out",
            "#SBATCH --error=%x-%j.err",
            "#SBATCH --hint=nomultithread",
        ]
        if cfg.partition:
            directives.append(f"#SBATCH --partition={cfg.partition}")
        if cfg.exclude:
            directives.append(f"#SBATCH --exclude={cfg.exclude}")
        if walltime:
            directives.append(f"#SBATCH --time={walltime}")
        directives.extend(f"#SBATCH {extra}" for extra in cfg.sbatch_extra)

        launch = cfg.launcher.format(
            binary=binary or cfg.opensees_bin, deck=deck, np=np
        )
        # sbatch propagates the *submission* environment (--export=ALL), and
        # ours is a non-interactive ssh shell that never sourced the login
        # PATH — so `srun` inside the script needs slurm_bin appended here.
        path_fix = (
            [f'export PATH="$PATH:{cfg.slurm_bin.rstrip("/")}"'] if cfg.slurm_bin else []
        )
        lines = [
            "#!/bin/bash",
            *directives,
            "",
            "set -uo pipefail",
            "",
            'echo "PWD: $(pwd)"; echo "HOST: $(hostname)"; date',
            *path_fix,
            *cfg.env,
            "",
            "EXIT_CODE=0",
            f"{launch} || EXIT_CODE=$?",
            "",
            'echo "$EXIT_CODE" > .exit_code',
            'echo "Elapsed: $SECONDS seconds."',
            'echo "LARGA VIDA AL LADRUÑO!!!"',
            'exit "$EXIT_CODE"',
        ]
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------- submit
    def submit(
        self,
        job_dir: str | Path,
        *,
        np: int | None = None,
        name: str | None = None,
        deck: str = "main.tcl",
        binary: str | None = None,
        walltime: str | None = None,
        overwrite: bool = False,
    ) -> Job:
        """Push ``job_dir`` to the cluster and sbatch it. Returns immediately.

        Parameters
        ----------
        job_dir:
            Local directory holding the deck (and everything it sources).
            Pushed wholesale to ``<remote_root>/<name>/``.
        np:
            MPI ranks (``--ntasks``). Falls back to the config's
            ``default_np``; required if neither is set. For a partitioned
            apeGmsh deck this is the partition count.
        name:
            Job name; defaults to the directory name. Also names the remote
            directory and the ``<name>-<jobid>.out/.err`` logs.
        deck:
            Entry script, relative to ``job_dir``.
        binary:
            Override the config's ``opensees_bin`` for this job (clusters
            keep several dated builds).
        overwrite:
            A remote ``<remote_root>/<name>/`` that already exists fails
            loud by default; ``True`` removes it first.
        """
        cfg = self.config
        local = Path(job_dir)
        if not local.is_dir():
            raise FileNotFoundError(f"job directory not found: {local}")
        if not (local / deck).is_file():
            raise FileNotFoundError(f"deck {deck!r} not found in {local}")

        job_name = name if name is not None else local.resolve().name
        if not _NAME_RE.match(job_name):
            raise ValueError(
                f"job name {job_name!r} must match {_NAME_RE.pattern} "
                "(it becomes a remote path and an sbatch directive); "
                "pass name=... to override the directory-derived default"
            )
        ranks = np if np is not None else cfg.default_np
        if not ranks or ranks < 1:
            raise ValueError("np must be a positive rank count (or set default_np)")

        script = self.render_batch_script(
            name=job_name, np=ranks, deck=deck, binary=binary, walltime=walltime
        )
        # LF + UTF-8 explicitly: a CRLF-authored script breaks remote bash.
        (local / "job.sbatch").write_bytes(script.encode("utf-8"))

        remote_dir = str(PurePosixPath(cfg.remote_root) / job_name)
        exists = _ssh.ssh(cfg.ssh_host, f"test -e {remote_dir}", check=False)
        if exists.returncode == 0:
            if not overwrite:
                raise HPCError(
                    f"remote job directory already exists: {cfg.ssh_host}:{remote_dir} "
                    "(pass overwrite=True to replace it, or pick another name=)"
                )
            _ssh.ssh(cfg.ssh_host, f"rm -rf {remote_dir}")

        _ssh.push_dir(cfg.ssh_host, local, remote_dir)

        proc = _ssh.ssh(
            cfg.ssh_host, f"cd {remote_dir} && {cfg.slurm('sbatch')} job.sbatch"
        )
        match = _SBATCH_ID_RE.search(proc.stdout)
        if not match:
            raise HPCError(
                f"could not parse sbatch output: {proc.stdout!r} "
                f"(stderr: {proc.stderr.strip()!r})"
            )

        job = Job(
            cluster=self,
            name=job_name,
            slurm_id=int(match.group(1)),
            remote_dir=remote_dir,
            local_dir=local,
            deck=deck,
        )
        job.save()
        return job
