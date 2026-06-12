"""Fixtures for apeGmsh.hpc — everything external is faked at the
``_ssh.run_local`` seam, so no test here ever touches ssh/scp/tar."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from apeGmsh.hpc import _ssh
from apeGmsh.hpc._config import ClusterConfig
from apeGmsh.hpc._cluster import Cluster

CLUSTERS_TOML = """\
[testcluster]
ssh_host = "testhost"
remote_root = "/remote/jobs"
opensees_bin = "/remote/bin/opensees"
slurm_bin = "/opt/slurm/bin"
partition = "computes"
exclude = "node17,node18"
env = ["export OMP_NUM_THREADS=1", "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/lib"]
"""


class FakeRun:
    """Scripted stand-in for ``_ssh.run_local``.

    Handlers are (substring, response) pairs matched against the joined
    argv, first match wins; unmatched commands succeed silently. Mirrors
    the real seam's ``check=`` semantics so error paths are honest.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._handlers: list[tuple[str, subprocess.CompletedProcess[str]]] = []

    def on(self, substring: str, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self._handlers.append(
            (substring, subprocess.CompletedProcess([], returncode, stdout, stderr))
        )

    def joined_calls(self) -> list[str]:
        return [" ".join(argv) for argv in self.calls]

    def __call__(
        self, argv: list[str], *, check: bool = True, timeout: float = 600.0
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        joined = " ".join(argv)
        resp = subprocess.CompletedProcess(argv, 0, "", "")
        for substring, scripted in self._handlers:
            if substring in joined:
                resp = subprocess.CompletedProcess(
                    argv, scripted.returncode, scripted.stdout, scripted.stderr
                )
                break
        if check and resp.returncode != 0:
            raise _ssh.HPCError(
                f"command failed (exit {resp.returncode}): {joined}\n"
                f"stderr: {resp.stderr.strip()}"
            )
        return resp


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(_ssh, "run_local", fake)
    return fake


@pytest.fixture
def clusters_toml(tmp_path: Path) -> Path:
    path = tmp_path / "clusters.toml"
    path.write_text(CLUSTERS_TOML, encoding="utf-8")
    return path


@pytest.fixture
def config(clusters_toml: Path) -> ClusterConfig:
    return ClusterConfig.load("testcluster", path=clusters_toml)


@pytest.fixture
def cluster(config: ClusterConfig) -> Cluster:
    return Cluster(config)


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "my-job"
    d.mkdir()
    (d / "main.tcl").write_text("puts ok\n", encoding="utf-8")
    return d
