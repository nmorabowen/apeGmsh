"""Cluster.submit — push sequence, sbatch parsing, sidecar, guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.hpc import HPCError
from apeGmsh.hpc._cluster import Cluster
from apeGmsh.hpc._job import SIDECAR_NAME

from tests.hpc.conftest import FakeRun


def _arm_happy_path(fake_run: FakeRun, *, job_id: int = 143701) -> None:
    fake_run.on("test -e", returncode=1)  # remote dir does not exist
    fake_run.on("sbatch", stdout=f"Submitted batch job {job_id}\n")


class TestSubmit:
    def test_happy_path(self, cluster: Cluster, fake_run: FakeRun, job_dir: Path) -> None:
        _arm_happy_path(fake_run)
        job = cluster.submit(job_dir, np=8)

        assert job.slurm_id == 143701
        assert job.name == "my-job"
        assert job.remote_dir == "/remote/jobs/my-job"
        assert job.local_dir == job_dir
        assert job.deck == "main.tcl"

    def test_command_sequence(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        _arm_happy_path(fake_run)
        cluster.submit(job_dir, np=8)
        joined = fake_run.joined_calls()

        # existence guard -> tar -> mkdir -> scp -> untar -> sbatch
        assert "test -e /remote/jobs/my-job" in joined[0]
        assert joined[1].startswith("tar -czf")
        assert "mkdir -p /remote/jobs/my-job" in joined[2]
        assert joined[3].startswith("scp")
        assert "testhost:/remote/jobs/my-job/.apegmsh_push.tgz" in joined[3]
        assert "tar -xzf /remote/jobs/my-job/.apegmsh_push.tgz" in joined[4]
        assert (
            "cd /remote/jobs/my-job && /opt/slurm/bin/sbatch job.sbatch" in joined[5]
        )
        # every remote command goes through non-interactive ssh
        for call in (joined[0], joined[2], joined[4], joined[5]):
            assert "BatchMode=yes" in call

    def test_writes_batch_script_lf_utf8(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        _arm_happy_path(fake_run)
        cluster.submit(job_dir, np=8)
        raw = (job_dir / "job.sbatch").read_bytes()
        assert b"\r" not in raw
        assert "LARGA VIDA AL LADRUÑO!!!".encode() in raw

    def test_writes_sidecar(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        _arm_happy_path(fake_run)
        cluster.submit(job_dir, np=8)
        payload = json.loads((job_dir / SIDECAR_NAME).read_text(encoding="utf-8"))
        assert payload == {
            "cluster": "testcluster",
            "config_path": cluster.config.config_path,
            "name": "my-job",
            "slurm_id": 143701,
            "remote_dir": "/remote/jobs/my-job",
            "deck": "main.tcl",
        }

    def test_default_np_fallback(self, fake_run: FakeRun, job_dir: Path) -> None:
        from apeGmsh.hpc._config import ClusterConfig

        cfg = ClusterConfig.from_dict(
            "c",
            {
                "ssh_host": "h",
                "remote_root": "/r",
                "opensees_bin": "/b",
                "default_np": 4,
            },
        )
        _arm_happy_path(fake_run)
        job = Cluster(cfg).submit(job_dir, name="j")
        assert "#SBATCH --ntasks=4" in (job_dir / "job.sbatch").read_text(
            encoding="utf-8"
        )
        assert job.slurm_id == 143701


class TestGuards:
    def test_missing_dir_fails(self, cluster: Cluster, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="job directory"):
            cluster.submit(tmp_path / "absent", np=2)

    def test_missing_deck_fails(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="deck 'other.tcl'"):
            cluster.submit(job_dir, np=2, deck="other.tcl")
        assert fake_run.calls == []  # failed before any remote contact

    def test_bad_name_fails(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="job name"):
            cluster.submit(job_dir, np=2, name="has spaces")
        assert fake_run.calls == []

    def test_missing_np_fails(self, cluster: Cluster, job_dir: Path) -> None:
        with pytest.raises(ValueError, match="np must be"):
            cluster.submit(job_dir)

    def test_existing_remote_dir_fails_loud(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        fake_run.on("test -e", returncode=0)  # exists
        with pytest.raises(HPCError, match="already exists"):
            cluster.submit(job_dir, np=2)

    def test_overwrite_removes_remote_dir(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        fake_run.on("test -e", returncode=0)
        fake_run.on("sbatch", stdout="Submitted batch job 7\n")
        job = cluster.submit(job_dir, np=2, overwrite=True)
        assert job.slurm_id == 7
        assert any("rm -rf /remote/jobs/my-job" in c for c in fake_run.joined_calls())

    def test_unparseable_sbatch_output_fails(
        self, cluster: Cluster, fake_run: FakeRun, job_dir: Path
    ) -> None:
        fake_run.on("test -e", returncode=1)
        fake_run.on("sbatch", stdout="something unexpected\n")
        with pytest.raises(HPCError, match="could not parse sbatch output"):
            cluster.submit(job_dir, np=2)
