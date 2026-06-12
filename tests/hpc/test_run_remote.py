"""ops.run_remote — the ADR 0060 bridge sugar (emit -> submit -> wait -> fetch).

Drives a real apeSees bridge over the fem stub; everything external is faked
at the ``_ssh.run_local`` seam, so the deck emit is real and the cluster is
not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.hpc import HPCError, Job
from apeGmsh.hpc._cluster import Cluster
from apeGmsh.opensees import apeSees

from tests.hpc.conftest import FakeRun
from tests.opensees.fixtures.fem_stub import make_two_node_beam


@pytest.fixture
def ops() -> apeSees:
    ops = apeSees(make_two_node_beam())
    ops.model(ndm=3, ndf=6)
    return ops


def _arm_full_lifecycle(fake_run: FakeRun, *, exit_code: str = "0") -> None:
    fake_run.on("test -e", returncode=1)
    fake_run.on("sbatch", stdout="Submitted batch job 555\n")
    fake_run.on("squeue", stdout="")  # terminal immediately
    fake_run.on(".exit_code", stdout=f"{exit_code}\n")


class TestRunRemote:
    def test_full_loop(
        self,
        ops: apeSees,
        cluster: Cluster,
        fake_run: FakeRun,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _arm_full_lifecycle(fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)
        job_dir = tmp_path / "remote-job"

        job = ops.run_remote(str(job_dir), cluster=cluster, np=2)

        assert isinstance(job, Job)
        assert job.slurm_id == 555
        # the deck was really emitted by the bridge before the push
        deck = (job_dir / "main.tcl").read_text(encoding="utf-8")
        assert "model BasicBuilder -ndm 3 -ndf 6" in deck
        # full lifecycle hit the seam: push, sbatch, squeue, sentinel, pull
        joined = " ".join(fake_run.joined_calls())
        assert "sbatch" in joined
        assert ".exit_code" in joined
        assert "apegmsh_pull" in joined  # fetch happened

    def test_np_defaults_to_partition_count_or_one(
        self,
        ops: apeSees,
        cluster: Cluster,
        fake_run: FakeRun,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _arm_full_lifecycle(fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)
        job_dir = tmp_path / "flat-job"

        ops.run_remote(str(job_dir), cluster=cluster)  # flat stub -> np=1

        script = (job_dir / "job.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --ntasks=1" in script

    def test_wait_false_returns_submitted_job_without_fetch(
        self,
        ops: apeSees,
        cluster: Cluster,
        fake_run: FakeRun,
        tmp_path: Path,
    ) -> None:
        fake_run.on("test -e", returncode=1)
        fake_run.on("sbatch", stdout="Submitted batch job 7\n")

        job = ops.run_remote(str(tmp_path / "j"), cluster=cluster, np=2, wait=False)

        assert job.slurm_id == 7
        joined = " ".join(fake_run.joined_calls())
        assert "squeue" not in joined
        assert "apegmsh_pull" not in joined

    def test_failed_job_fetches_then_raises_with_stderr_tail(
        self,
        ops: apeSees,
        cluster: Cluster,
        fake_run: FakeRun,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _arm_full_lifecycle(fake_run, exit_code="137")
        fake_run.on("tail", stdout="boom: segfault in rank 1\n")
        monkeypatch.setattr("time.sleep", lambda s: None)

        with pytest.raises(HPCError, match=r"ended FAILED[\s\S]*segfault in rank 1"):
            ops.run_remote(str(tmp_path / "j"), cluster=cluster, np=2)
        # evidence was fetched BEFORE the raise
        assert any("apegmsh_pull" in c for c in fake_run.joined_calls())

    def test_cluster_by_name_uses_config_loader(
        self,
        ops: apeSees,
        fake_run: FakeRun,
        clusters_toml: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from apeGmsh.hpc import _config

        monkeypatch.setattr(_config, "DEFAULT_CONFIG_PATH", clusters_toml)
        _arm_full_lifecycle(fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)

        job = ops.run_remote(
            str(tmp_path / "j"), cluster="testcluster", np=2
        )
        assert job.cluster.config.ssh_host == "testhost"
