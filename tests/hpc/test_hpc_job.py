"""Job — status resolution (squeue + sentinel), tail, cancel, fetch, reload."""

from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.hpc import Job, JobStatus
from apeGmsh.hpc._cluster import Cluster

from tests.hpc.conftest import FakeRun


@pytest.fixture
def job(cluster: Cluster, job_dir: Path) -> Job:
    return Job(
        cluster=cluster,
        name="my-job",
        slurm_id=143701,
        remote_dir="/remote/jobs/my-job",
        local_dir=job_dir,
        deck="main.tcl",
    )


class TestStatus:
    @pytest.mark.parametrize(
        ("squeue_state", "expected"),
        [
            ("PENDING", JobStatus.PENDING),
            ("CONFIGURING", JobStatus.PENDING),
            ("RUNNING", JobStatus.RUNNING),
            ("COMPLETING", JobStatus.RUNNING),
            ("CANCELLED", JobStatus.CANCELLED),
        ],
    )
    def test_live_states_from_squeue(
        self, job: Job, fake_run: FakeRun, squeue_state: str, expected: JobStatus
    ) -> None:
        fake_run.on("squeue", stdout=f"{squeue_state}\n")
        assert job.status() is expected
        assert "/opt/slurm/bin/squeue -h -j 143701 -o %T" in fake_run.joined_calls()[0]

    def test_finished_job_reads_exit_sentinel_zero(
        self, job: Job, fake_run: FakeRun
    ) -> None:
        # sacct may be broken (e.g. Esmeralda); the sentinel is the contract.
        fake_run.on("squeue", returncode=1, stderr="Invalid job id specified")
        fake_run.on("cat /remote/jobs/my-job/.exit_code", stdout="0\n")
        assert job.status() is JobStatus.COMPLETED

    def test_finished_job_nonzero_sentinel_is_failed(
        self, job: Job, fake_run: FakeRun
    ) -> None:
        fake_run.on("squeue", stdout="")
        fake_run.on("cat /remote/jobs/my-job/.exit_code", stdout="137\n")
        assert job.status() is JobStatus.FAILED

    def test_no_queue_entry_and_no_sentinel_is_unknown(
        self, job: Job, fake_run: FakeRun
    ) -> None:
        fake_run.on("squeue", returncode=1)
        fake_run.on(".exit_code", returncode=1, stderr="No such file or directory")
        assert job.status() is JobStatus.UNKNOWN

    def test_terminal_predicate(self) -> None:
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal
        assert not JobStatus.RUNNING.is_terminal
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.UNKNOWN.is_terminal


class TestWait:
    def test_polls_until_terminal(
        self, job: Job, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        from apeGmsh.hpc import _ssh

        # squeue answers RUNNING twice, then the job leaves the queue and
        # the sentinel says success.
        states = iter(["RUNNING\n", "RUNNING\n", ""])

        def fake_call(argv, *, check=True, timeout=600.0):
            joined = " ".join(argv)
            if "squeue" in joined:
                return sp.CompletedProcess(argv, 0, next(states), "")
            if ".exit_code" in joined:
                return sp.CompletedProcess(argv, 0, "0\n", "")
            return sp.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(_ssh, "run_local", fake_call)
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

        assert job.wait(poll=5.0) is JobStatus.COMPLETED
        assert sleeps == [5.0, 5.0]

    def test_timeout_raises_and_job_keeps_running(
        self, job: Job, fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_run.on("squeue", stdout="RUNNING\n")
        clock = iter([0.0, 100.0, 200.0])
        monkeypatch.setattr("time.monotonic", lambda: next(clock))
        monkeypatch.setattr("time.sleep", lambda s: None)
        with pytest.raises(TimeoutError, match="still RUNNING"):
            job.wait(poll=1.0, timeout=50.0)


class TestTailCancel:
    def test_tail_targets_named_log(self, job: Job, fake_run: FakeRun) -> None:
        fake_run.on("tail", stdout="last lines\n")
        out = job.tail(20)
        assert out == "last lines\n"
        assert (
            "tail -n 20 /remote/jobs/my-job/my-job-143701.out"
            in fake_run.joined_calls()[0]
        )

    def test_tail_err_stream(self, job: Job, fake_run: FakeRun) -> None:
        job.tail(stream="err")
        assert "my-job-143701.err" in fake_run.joined_calls()[0]

    def test_tail_invalid_stream(self, job: Job) -> None:
        with pytest.raises(ValueError, match="stream"):
            job.tail(stream="log")

    def test_tail_missing_log_is_soft(self, job: Job, fake_run: FakeRun) -> None:
        fake_run.on("tail", returncode=1, stderr="No such file")
        assert "no out log yet" in job.tail()

    def test_cancel(self, job: Job, fake_run: FakeRun) -> None:
        job.cancel()
        assert "/opt/slurm/bin/scancel 143701" in fake_run.joined_calls()[0]


class TestFetch:
    def test_fetch_pulls_into_local_dir(self, job: Job, fake_run: FakeRun) -> None:
        dest = job.fetch()
        assert dest == job.local_dir
        joined = fake_run.joined_calls()
        # remote tar (staged OUTSIDE the job dir) -> scp back -> remote
        # cleanup -> local untar
        assert "tar -czf /tmp/apegmsh_pull_my-job.tgz -C /remote/jobs/my-job" in joined[0]
        assert joined[1].startswith("scp")
        assert "testhost:/tmp/apegmsh_pull_my-job.tgz" in joined[1]
        assert "rm -f /tmp/apegmsh_pull_my-job.tgz" in joined[2]
        assert joined[3].startswith("tar -xzf")

    def test_fetch_explicit_dest(
        self, job: Job, fake_run: FakeRun, tmp_path: Path
    ) -> None:
        dest = job.fetch(tmp_path / "results")
        assert dest == tmp_path / "results"
        assert dest.is_dir()  # created locally before the pull


class TestSidecarRoundTrip:
    def test_save_then_load(self, job: Job) -> None:
        job.save()
        loaded = Job.load(job.local_dir)
        assert loaded.name == job.name
        assert loaded.slurm_id == job.slurm_id
        assert loaded.remote_dir == job.remote_dir
        assert loaded.deck == job.deck
        assert loaded.cluster.config == job.cluster.config

    def test_load_without_sidecar_fails(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="apegmsh_job.json"):
            Job.load(tmp_path)
