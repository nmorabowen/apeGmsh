# ADR 0060 — Remote HPC job submission (`apeGmsh.hpc`)

**Status:** Accepted — v1 implemented (Cluster/Job primitives; run-verified
end-to-end against the Esmeralda cluster, 2026-06-11).

## Context

apeGmsh emits partitioned OpenSees decks that run on a SLURM cluster
(Esmeralda: 18 × 16-core nodes, single `computes` partition). The manual
loop — scp the deck up, ssh in, `sbatch`, watch, scp results back — is
mechanical and was already half-automated by hand-rolled bash on the
cluster (`~/nmbUAndes/run.sh` + `runnn.sh`, which sed-renames a template
sbatch script and infers `--ntasks` by counting `*.part-*.mpco.cdata`
files). apeGmsh knows the partition count at emit time, so the inference
hack and the copy/rename dance can be owned by a typed Python surface.

Probed cluster facts that constrain the design:

- SLURM lives at `/opt/slurm/bin`, which is **only on the login-shell
  PATH**. Non-interactive `ssh host cmd` never sources `/etc/profile`, so
  the tool must call SLURM by absolute path (config `slurm_bin`).
- `MpiDefault=pmix_v3`; the proven launch line is
  `srun --cpu-bind=cores --mpi=pmix_v3 <binary> <deck>` — **not** mpiexec.
- **`sacct` is broken** (slurmdbd down, `JobAcctGatherType=none`). A
  finished job vanishes from `squeue` with no queryable history.
- Windows client: OpenSSH and bsdtar ship with the OS; rsync does not.

## Decision

A standalone `apeGmsh.hpc` module owning **push → sbatch → poll → fetch**,
deliberately *not* coupled to the bridge (the deck is emitted by the caller
as usual). Bridge sugar (`p.run_remote(...)`) can wrap these primitives
later once proven.

1. **Two-layer config.** `~/.ssh/config` owns the connection (alias, port,
   user, key) — apeGmsh never touches credentials. `~/.apegmsh/clusters.toml`
   owns cluster facts only (`remote_root`, `slurm_bin`, `partition`,
   `opensees_bin`, `env`, `launcher`, `exclude`, `sbatch_extra`,
   `default_np`), loaded into a frozen `ClusterConfig` with fail-loud
   unknown-key / missing-key / whitespace-in-path validation.
2. **Native-subprocess transport.** All external commands funnel through a
   single seam (`_ssh.run_local`) invoking the OS `ssh`/`scp`/`tar` with
   `BatchMode=yes` (fail fast, never prompt). Zero new Python dependencies
   (`tomllib`; `tomli` on 3.10). Directory transfer is one gzipped tarball
   each way — one round-trip regardless of file count, no rsync needed.
3. **Generated, inspectable batch script.** `Cluster.submit(job_dir, np=…)`
   renders `job.sbatch` (LF + UTF-8 explicitly — CRLF breaks remote bash)
   *into the local job dir* before pushing, so what ran always sits next to
   the deck. Remote staging is `remote_root/<name>/`; an existing remote
   dir fails loud unless `overwrite=True`.
4. **Exit-code sentinel instead of accounting.** The script's last act is
   `echo "$EXIT_CODE" > .exit_code`. `Job.status()` resolves from `squeue`
   while the job is alive and falls back to the sentinel once it leaves the
   queue (`COMPLETED`/`FAILED`/`UNKNOWN`). This is the only durable
   completion record on a cluster without working `sacct`.
5. **Session-surviving handle.** `submit()` writes a JSON sidecar
   (`.apegmsh_job.json`) into the local job dir; `Job.load(dir)` rehydrates
   the handle (cluster resolved by name from the TOML) for later
   `status()`/`tail()`/`cancel()`/`fetch()`.

## Consequences

- Job names are restricted to `[A-Za-z0-9._-]+` and config paths must be
  whitespace-free: remote commands are interpolated strings by design (no
  quoting layer). Fail-loud guards enforce both.
- `np` maps 1:1 to `--ntasks` and to the deck's partition count; the
  `.cdata`-counting inference of `runnn.sh` is retired on this path.
- The launcher template (`{binary}` / `{deck}` / `{np}`) keeps the module
  scheduler-recipe-agnostic; a plain-mpiexec cluster is a one-line config
  change away.
- Unit tests fake the single subprocess seam (no network in CI); the live
  smoke (`APEGMSH_HPC_LIVE=<cluster>`) is opt-in.
- Deferred: `p.run_remote` bridge sugar, `job.wait()` convenience,
  memory-tracker opt-in from the legacy `run.sh`, multi-job folder sweeps
  (`run_folder_v*` parity), and results-side `Results.from_…` auto-open of
  fetched directories.
