"""ClusterConfig — TOML loading, defaults, fail-loud validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.hpc import ClusterConfig
from apeGmsh.hpc._config import DEFAULT_LAUNCHER


class TestLoad:
    def test_loads_named_block(self, clusters_toml: Path) -> None:
        cfg = ClusterConfig.load("testcluster", path=clusters_toml)
        assert cfg.name == "testcluster"
        assert cfg.ssh_host == "testhost"
        assert cfg.remote_root == "/remote/jobs"
        assert cfg.opensees_bin == "/remote/bin/opensees"
        assert cfg.partition == "computes"
        assert cfg.exclude == "node17,node18"
        assert cfg.env == (
            "export OMP_NUM_THREADS=1",
            "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/lib",
        )
        assert cfg.config_path == str(clusters_toml)

    def test_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text(
            '[mini]\nssh_host = "h"\nremote_root = "/r"\nopensees_bin = "/b"\n',
            encoding="utf-8",
        )
        cfg = ClusterConfig.load("mini", path=path)
        assert cfg.slurm_bin == ""
        assert cfg.partition == ""
        assert cfg.launcher == DEFAULT_LAUNCHER
        assert cfg.env == ()
        assert cfg.sbatch_extra == ()
        assert cfg.default_np == 0

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config file not found"):
            ClusterConfig.load("x", path=tmp_path / "absent.toml")

    def test_unknown_cluster_lists_available(self, clusters_toml: Path) -> None:
        with pytest.raises(ValueError, match=r"available: testcluster"):
            ClusterConfig.load("nope", path=clusters_toml)


class TestValidation:
    def test_missing_required_key_fails(self) -> None:
        with pytest.raises(ValueError, match=r"required key.*'opensees_bin'"):
            ClusterConfig.from_dict("c", {"ssh_host": "h", "remote_root": "/r"})

    def test_unknown_key_fails(self) -> None:
        with pytest.raises(ValueError, match="unknown config key"):
            ClusterConfig.from_dict(
                "c",
                {
                    "ssh_host": "h",
                    "remote_root": "/r",
                    "opensees_bin": "/b",
                    "partion": "typo",
                },
            )

    def test_whitespace_in_remote_path_fails(self) -> None:
        with pytest.raises(ValueError, match="whitespace"):
            ClusterConfig.from_dict(
                "c",
                {"ssh_host": "h", "remote_root": "/r oot", "opensees_bin": "/b"},
            )


class TestSlurmPath:
    def test_joins_slurm_bin(self, config: ClusterConfig) -> None:
        assert config.slurm("sbatch") == "/opt/slurm/bin/sbatch"
        assert config.slurm("squeue") == "/opt/slurm/bin/squeue"

    def test_empty_slurm_bin_means_path(self) -> None:
        cfg = ClusterConfig.from_dict(
            "c", {"ssh_host": "h", "remote_root": "/r", "opensees_bin": "/b"}
        )
        assert cfg.slurm("sbatch") == "sbatch"
