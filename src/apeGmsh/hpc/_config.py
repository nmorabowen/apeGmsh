"""Cluster configuration — the TOML half of the two-layer config contract.

The SSH connection (host, port, user, key) is owned by ``~/.ssh/config``;
this file owns only the *cluster facts* SSH cannot know: where jobs stage,
where SLURM and OpenSees live, and how a deck is launched. Keeping the two
layers separate means apeGmsh never re-implements (or stores) credentials —
``ssh_host`` is just an alias the native ``ssh.exe`` resolves.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "apeGmsh.hpc needs a TOML parser: Python >= 3.11 (stdlib tomllib) "
            "or `pip install tomli` on 3.10."
        ) from exc

DEFAULT_CONFIG_PATH = Path.home() / ".apegmsh" / "clusters.toml"

#: Default launch line inside the batch script. ``{binary}`` / ``{deck}`` /
#: ``{np}`` are substituted at render time. ``srun`` (not ``mpiexec``) is the
#: SLURM-native MPI launcher; ``pmix_v3`` matches a ``MpiDefault=pmix_v3``
#: cluster (Esmeralda's), override per cluster otherwise.
DEFAULT_LAUNCHER = "srun --cpu-bind=cores --mpi=pmix_v3 {binary} {deck}"


@dataclass(frozen=True)
class ClusterConfig:
    """Frozen facts about one cluster, loaded from ``clusters.toml``.

    Required keys: ``ssh_host``, ``remote_root``, ``opensees_bin``.
    Everything else has a working default. Unknown keys in the TOML block
    fail loud (typo protection).
    """

    name: str
    ssh_host: str
    remote_root: str
    opensees_bin: str
    #: Directory holding sbatch/squeue/scancel. Empty string = rely on PATH.
    #: Needed when SLURM is only on the *login-shell* PATH (e.g. Esmeralda's
    #: ``/opt/slurm/bin``) — non-interactive ``ssh host cmd`` never sources
    #: ``/etc/profile``, so absolute paths are the reliable route.
    slurm_bin: str = ""
    partition: str = ""
    launcher: str = DEFAULT_LAUNCHER
    #: Shell lines run in the batch script before the launcher
    #: (module loads, OMP_NUM_THREADS, LD_LIBRARY_PATH, ...).
    env: tuple[str, ...] = ()
    #: Nodes to exclude, e.g. ``"node17,node18"``. Empty = none.
    exclude: str = ""
    #: Extra raw sbatch options, one per entry, e.g. ``"--mem=8G"``.
    sbatch_extra: tuple[str, ...] = ()
    default_np: int = 0
    config_path: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for key in ("ssh_host", "remote_root", "opensees_bin"):
            value = getattr(self, key)
            if not value or not str(value).strip():
                raise ValueError(
                    f"cluster {self.name!r}: required key {key!r} is missing or empty"
                )
        # Remote paths are interpolated into ssh command strings; whitespace
        # would need a quoting layer this module deliberately does not have.
        for key in ("remote_root", "opensees_bin", "slurm_bin"):
            value = str(getattr(self, key))
            if any(ch.isspace() for ch in value):
                raise ValueError(
                    f"cluster {self.name!r}: {key!r} must not contain whitespace, "
                    f"got {value!r}"
                )

    @classmethod
    def load(cls, name: str, path: str | Path | None = None) -> ClusterConfig:
        """Load the ``[name]`` block from ``clusters.toml``.

        ``path`` defaults to ``~/.apegmsh/clusters.toml``.
        """
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            raise FileNotFoundError(
                f"cluster config file not found: {config_path} "
                "(create it with one [cluster-name] block per cluster)"
            )
        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
        if name not in data:
            available = ", ".join(sorted(data)) or "<none>"
            raise ValueError(
                f"cluster {name!r} not found in {config_path} (available: {available})"
            )
        return cls.from_dict(name, data[name], config_path=str(config_path))

    @classmethod
    def from_dict(
        cls, name: str, raw: dict[str, Any], *, config_path: str = ""
    ) -> ClusterConfig:
        known = {f.name for f in fields(cls)} - {"name", "config_path"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"cluster {name!r}: unknown config key(s) {sorted(unknown)} "
                f"(known: {sorted(known)})"
            )
        missing = {"ssh_host", "remote_root", "opensees_bin"} - set(raw)
        if missing:
            raise ValueError(
                f"cluster {name!r}: missing required key(s): "
                + ", ".join(repr(k) for k in sorted(missing))
            )
        kwargs: dict[str, Any] = dict(raw)
        for key in ("env", "sbatch_extra"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(name=name, config_path=config_path, **kwargs)

    def slurm(self, tool: str) -> str:
        """Absolute (or bare, if ``slurm_bin`` is empty) path to a SLURM tool."""
        if self.slurm_bin:
            return f"{self.slurm_bin.rstrip('/')}/{tool}"
        return tool
