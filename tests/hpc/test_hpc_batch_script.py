"""Batch-script rendering — directives, launcher substitution, sentinel."""

from __future__ import annotations

import pytest

from apeGmsh.hpc._cluster import Cluster
from apeGmsh.hpc._config import ClusterConfig


@pytest.fixture
def script(cluster: Cluster) -> str:
    return cluster.render_batch_script(name="myjob", np=8, deck="main.tcl")


class TestDirectives:
    def test_core_directives(self, script: str) -> None:
        assert script.startswith("#!/bin/bash\n")
        assert "#SBATCH --job-name=myjob" in script
        assert "#SBATCH --ntasks=8" in script
        assert "#SBATCH --cpus-per-task=1" in script
        assert "#SBATCH --output=%x-%j.out" in script
        assert "#SBATCH --error=%x-%j.err" in script
        assert "#SBATCH --hint=nomultithread" in script

    def test_config_driven_directives(self, script: str) -> None:
        assert "#SBATCH --partition=computes" in script
        assert "#SBATCH --exclude=node17,node18" in script

    def test_optional_directives_omitted_when_unset(self) -> None:
        cfg = ClusterConfig.from_dict(
            "c", {"ssh_host": "h", "remote_root": "/r", "opensees_bin": "/b"}
        )
        script = Cluster(cfg).render_batch_script(name="j", np=2, deck="d.tcl")
        assert "--partition" not in script
        assert "--exclude" not in script
        assert "--time" not in script

    def test_walltime(self, cluster: Cluster) -> None:
        script = cluster.render_batch_script(
            name="j", np=2, deck="d.tcl", walltime="04:00:00"
        )
        assert "#SBATCH --time=04:00:00" in script

    def test_sbatch_extra(self) -> None:
        cfg = ClusterConfig.from_dict(
            "c",
            {
                "ssh_host": "h",
                "remote_root": "/r",
                "opensees_bin": "/b",
                "sbatch_extra": ["--mem=8G", "--ntasks-per-node=16"],
            },
        )
        script = Cluster(cfg).render_batch_script(name="j", np=2, deck="d.tcl")
        assert "#SBATCH --mem=8G" in script
        assert "#SBATCH --ntasks-per-node=16" in script


class TestBody:
    def test_launcher_substitution(self, script: str) -> None:
        assert (
            "srun --cpu-bind=cores --mpi=pmix_v3 /remote/bin/opensees main.tcl"
            " || EXIT_CODE=$?" in script
        )

    def test_binary_override(self, cluster: Cluster) -> None:
        script = cluster.render_batch_script(
            name="j", np=2, deck="d.tcl", binary="/remote/bin/opensees-new"
        )
        assert "/remote/bin/opensees-new d.tcl" in script
        assert "/remote/bin/opensees " not in script

    def test_np_substitution_in_custom_launcher(self) -> None:
        cfg = ClusterConfig.from_dict(
            "c",
            {
                "ssh_host": "h",
                "remote_root": "/r",
                "opensees_bin": "/b",
                "launcher": "mpiexec -np {np} {binary} {deck}",
            },
        )
        script = Cluster(cfg).render_batch_script(name="j", np=4, deck="d.tcl")
        assert "mpiexec -np 4 /b d.tcl || EXIT_CODE=$?" in script

    def test_slurm_bin_appended_to_path(self, script: str) -> None:
        # sbatch exports the submission env; a non-interactive ssh submission
        # has no login PATH, so srun inside the script needs this line.
        assert 'export PATH="$PATH:/opt/slurm/bin"' in script

    def test_no_path_line_without_slurm_bin(self) -> None:
        cfg = ClusterConfig.from_dict(
            "c", {"ssh_host": "h", "remote_root": "/r", "opensees_bin": "/b"}
        )
        script = Cluster(cfg).render_batch_script(name="j", np=2, deck="d.tcl")
        assert "export PATH=" not in script

    def test_env_lines(self, script: str) -> None:
        assert "export OMP_NUM_THREADS=1" in script
        assert "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/lib" in script

    def test_exit_code_sentinel_is_written_and_propagated(self, script: str) -> None:
        # Load-bearing on clusters without sacct: the sentinel is the only
        # durable record of how the job ended (Job.status reads it).
        assert 'echo "$EXIT_CODE" > .exit_code' in script
        assert script.rstrip().endswith('exit "$EXIT_CODE"')

    def test_larga_vida(self, script: str) -> None:
        assert "LARGA VIDA AL LADRUÑO!!!" in script

    def test_lf_only(self, script: str) -> None:
        assert "\r" not in script
