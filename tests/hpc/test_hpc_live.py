"""Opt-in live smoke against a real cluster — read-only, no job submitted.

Run with ``APEGMSH_HPC_LIVE=<cluster-name>`` set (e.g. ``esmeralda``); needs
that cluster in ``~/.apegmsh/clusters.toml`` and a working ssh alias. Skipped
everywhere else (CI has no key).
"""

from __future__ import annotations

import os

import pytest

from apeGmsh.hpc import Cluster
from apeGmsh.hpc import _ssh

LIVE_CLUSTER = os.environ.get("APEGMSH_HPC_LIVE", "")

pytestmark = pytest.mark.skipif(
    not LIVE_CLUSTER, reason="set APEGMSH_HPC_LIVE=<cluster-name> to run"
)


def test_ping_and_slurm_reachable() -> None:
    cluster = Cluster.load(LIVE_CLUSTER)
    assert cluster.ping(), f"ssh {cluster.config.ssh_host} did not answer"
    proc = _ssh.ssh(
        cluster.config.ssh_host,
        f"{cluster.config.slurm('squeue')} --version",
        timeout=30.0,
    )
    assert proc.stdout.startswith("slurm"), proc.stdout
