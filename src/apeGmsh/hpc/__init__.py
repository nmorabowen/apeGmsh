"""Remote HPC job submission over SSH + SLURM (ADR 0058).

Submit apeGmsh-emitted OpenSees decks to a SLURM cluster from the local
machine, using the native ``ssh``/``scp`` executables (the user's existing
``~/.ssh/config`` aliases and key agent — apeGmsh never touches credentials).

Connection facts live in ``~/.ssh/config``; cluster facts (paths, partition,
launcher recipe) live in ``~/.apegmsh/clusters.toml``::

    [esmeralda]
    ssh_host     = "esmeralda"
    remote_root  = "/mnt/deadmanschest/nmorabowen/apegmsh_jobs"
    slurm_bin    = "/opt/slurm/bin"
    partition    = "computes"
    opensees_bin = "/mnt/nfshare/bin/opensees-14072025"
    env          = ["export OMP_NUM_THREADS=1",
                    "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/mnt/nfshare/lib"]

Typical session::

    from apeGmsh.hpc import Cluster

    cluster = Cluster.load("esmeralda")
    job = cluster.submit("./job_dir", np=8, name="tunnel-v3")  # push + sbatch
    job.status()          # PENDING / RUNNING / COMPLETED / FAILED
    job.tail()            # last lines of the remote stdout
    job.fetch()           # pull results back into ./job_dir

    # later, from a fresh session:
    from apeGmsh.hpc import Job
    job = Job.load("./job_dir")

The deck is the caller's responsibility (emit with the bridge as usual);
this module owns only *push -> sbatch -> poll -> fetch*.
"""

from ._config import ClusterConfig
from ._cluster import Cluster
from ._job import Job, JobStatus
from ._ssh import HPCError

__all__ = ["Cluster", "ClusterConfig", "HPCError", "Job", "JobStatus"]
