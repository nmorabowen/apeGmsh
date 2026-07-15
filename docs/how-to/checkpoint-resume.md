# Checkpoint & resume an analysis

Save the committed state of a running OpenSees analysis to disk and pick it up
later — in a fresh process, after a crash, or as the start of a new leg — using
OpenSees' `FE_Datastore` (`database` / `save` / `restore`). This is *flavor-3
exact resume*: the same model, continued from an exact committed state.

Reach for this when a transient run is long and you want a **recovery point**, or
when you want a trusted **benchmark point** to validate that a restart reproduces
an uninterrupted run.

!!! warning "This is not `g.save()` / model archival"
    `g.save()` / `fem.to_h5()` / `apeGmsh.from_h5(...)` persist the **model**
    (geometry, mesh, tags) so you can rebuild and re-run it — see
    [Save & reload a model](save-reload.md). They do **not** carry the *committed
    solver state* (nodal velocity/acceleration, committed material state, current
    time) needed to *continue* a transient. That is what `save`/`restore` below
    do.

!!! note "apeGmsh `checkpoint` / `from_checkpoint` is planned, not shipped"
    **ADR 0043** (`src/apeGmsh/opensees/architecture/decisions/0043-connectivity-graph-and-flexible-emit.md`,
    §*Restart surfaces*) designs first-class `res.checkpoint("tag")` /
    `apeSees.from_checkpoint("tag")`
    wrappers (FE_Datastore save **paired with a `model.h5`** for a deterministic
    topology rebuild). Until that lands, drive the underlying
    `ops.database`/`save`/`restore` on the live openseespy session directly, as
    shown here.

## Prerequisite: a build with the mass-restart fix

Restarting a **transient** run exercises the mass matrix. Continuum plane
elements (`quad`/`FourNodeQuad`, `tri31`, `NineNodeQuad`, `EightNodeQuad`,
`SixNodeTri`, `FourNodeQuad3d`) had an upstream bug where the element density was
not serialized, so a database/parallel restart rebuilt them with a **garbage
mass matrix** — non-deterministic across processes, with the effective operator's
eigenvalues going negative. This is fixed in Ladruno fork
**[PR #577](https://github.com/nmorabowen/OpenSees/pull/577)**.

Use a fork build that includes it. If unsure, run the
[verification benchmark](#step-3-benchmark-a-checkpoint-before-you-trust-it)
below: a build **without** the fix makes a restarted transient diverge wildly
(or throws on a negative-eigenvalue solve); a build **with** it reproduces the
uninterrupted run to machine precision.

## Step 1 — create a checkpoint

Build and run the first leg through apeGmsh exactly as usual, then reach the live
openseespy session apeGmsh just drove and write a datastore checkpoint:

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- build the model (mesh -> fem) as in any tutorial ---
# ... g = apeGmsh(...); ...; fem = g.mesh.queries.get_fem_data(dim=2)

ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
# ... declare materials / elements / fixes / a load pattern ...

# analysis chain
ops.constraints.Transformation()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=1e-8, max_iter=20)
ops.algorithm.Newton()
ops.integrator.Newmark(gamma=0.5, beta=0.25)
ops.analysis.Transient()

# leg 1: run up to the checkpoint
ops.analyze(steps=500, dt=0.01)

# --- write the checkpoint on the SAME live domain apeGmsh populated ---
try:
    import opensees as osp                # Ladruno fork (opensees.pyd)
except ModuleNotFoundError:
    import openseespy.opensees as osp     # stock openseespy

CKPT = r"C:\work\run42\ckpt\step500"      # LOCAL disk, no extension
osp.database("File", CKPT)
rc = osp.save(1)                          # 1 = the checkpoint key (commitTag)
assert rc is None or rc >= 0, "database save failed"
```

`save(k)` writes the committed domain — nodes (including committed
disp/**vel/accel**), elements and their committed material state, load patterns,
constraints, and the current pseudo-time — into a set of small files next to
`CKPT` (`step500.IDs.*`, `.VECs.*`, `.MATs.*`). `k` is an integer key you choose;
you can hold several checkpoints in one datastore under different keys.

Checkpoint **only at a committed state** — i.e. between `analyze` steps, never
mid-step. `ops.analyze(steps=N, dt=...)` leaves the domain committed at the last
step, which is exactly the right moment.

## Step 2 — resume from a checkpoint

Resume can run in a **fresh process**. `restore` rebuilds the *entire* model
(topology *and* committed state) straight from the datastore — you do **not**
rebuild the model through apeGmsh, and you must **not** call `ops.analyze()`
again (the apeGmsh live path wipes and re-emits the model from the mesh, which
would discard the restored state). Drive the continued analysis with raw
openseespy instead:

```python
try:
    import opensees as osp
except ModuleNotFoundError:
    import openseespy.opensees as osp

CKPT = r"C:\work\run42\ckpt\step500"

osp.wipe()                          # start from an empty domain
osp.database("File", CKPT)
osp.restore(1)                      # rebuilds nodes/elements/patterns + committed state

# re-declare the analysis chain (the datastore does NOT store it)
osp.constraints("Transformation")
osp.numberer("RCM")
osp.system("BandGeneral")
osp.test("NormDispIncr", 1e-8, 20)
osp.algorithm("Newton")
osp.integrator("Newmark", 0.5, 0.25)
osp.analysis("Transient")

# leg 2: continue exactly where leg 1 stopped
osp.analyze(500, 0.01)
```

The analysis chain (`constraints` … `analysis`) is **not** part of the saved
state — re-declare it after `restore`, matching the integrator you checkpointed
under. Everything else comes back from the datastore.

## Step 3 — benchmark a checkpoint before you trust it

Before you rely on a checkpoint, prove it reproduces the uninterrupted run. Run
the model straight through, then run it again with a `save`/`wipe`/`restore` in
the middle, and compare a probe over the *overlapping* steps:

```python
def leg(node, dof, steps, dt):
    out = []
    for _ in range(steps):
        assert osp.analyze(1, dt) == 0
        out.append(osp.nodeDisp(node, dof))
    return out

# A) uninterrupted reference: N steps straight through
build_and_setup_analysis()                      # your model + chain (raw osp)
ref = leg(TOP, 2, 12, 0.01)

# B) checkpoint at step 6, restore, continue to 12
build_and_setup_analysis()
first = leg(TOP, 2, 6, 0.01)
osp.database("File", CKPT); osp.save(1)
osp.wipe(); osp.database("File", CKPT); osp.restore(1)
setup_analysis_only()                           # re-declare the chain (no rebuild)
rest = leg(TOP, 2, 6, 0.01)

# steps 7..12 must match the reference to ~machine precision
for i, (a, b) in enumerate(zip(rest, ref[6:]), start=7):
    rel = abs(a - b) / (abs(b) + 1e-30)
    assert rel < 1e-9, f"step {i} diverged (rel={rel:.2e}) — bad build or un-restored state"
```

A correct build reproduces the reference to ~1 ULP (rel ≈ 1e-16); it will **not**
be bit-identical, because `restore` tears the domain down and rebuilds it, so
floating-point summation order in reassembly can differ by a bit — compare with a
tolerance, never with `==`. A restart that diverges by *percent* (or a negative
eigenvalue) means either an element that doesn't round-trip its state or a build
missing [PR #577](https://github.com/nmorabowen/OpenSees/pull/577).

For a stronger, cheaper probe of the mass operator, capture eigenvalues before
`save` and after `restore` and assert they match — a corrupt restored mass shows
up immediately as a changed (often negative) spectrum:

```python
before = sorted(osp.eigen("-fullGenLapack", 3))
# ... save / wipe / restore / re-declare chain ...
after  = sorted(osp.eigen("-fullGenLapack", 3))
assert all(abs(a - b) <= 1e-9 * abs(b) for a, b in zip(after, before))
```

## Notes / gotchas

- **The datastore is a *pile of small files*, not one file.** `database("File",
  CKPT)` writes `CKPT.IDs.*` / `CKPT.VECs.*` / `CKPT.MATs.*` next to `CKPT`. Point
  it at **local disk** — a network/SeaDrive path is slow and can corrupt or lock
  the many small handles. Keep each run's checkpoints in their own directory.
- **`restore` needs the SAME build.** The running OpenSees must register every
  element/material class tag in the checkpoint in its object broker. Restoring
  under a different (or older) build can fail with *"no Element type exists for
  class tag …"*. Checkpoint and resume with the same fork binary.
- **`save`/`restore` are all-or-nothing on the whole domain**, keyed by an integer
  `commitTag`. Reuse the same key you saved under; different keys hold different
  checkpoints in one datastore.
- **Re-`database` before `restore` in a reused process.** After `wipe()`, re-open
  the datastore (`osp.database("File", CKPT)`) before `restore` so the file
  handles are live; `wipe()` also releases handles on Windows before cleanup.
- **Staged models (`ops.stage(...)`) can't be driven live** (Phase SSI-2.A) — emit
  a deck with `ops.tcl(path, run=True)` / `ops.py(path, run=True)` and checkpoint
  from within the deck's Python instead.
- **Recorders don't resume themselves.** A recorder writes from when it is
  declared; re-declare your recorders after `restore` (and expect a fresh output
  file unless you manage appending yourself).

## See also

- [Save & reload a model](save-reload.md) — model archival (`g.save()` /
  `apeGmsh.from_h5`), the *other* kind of persistence.
- [Export to a Tcl / openseespy script](export-script.md) — for staged or
  deck-driven runs where you checkpoint from within the deck.
- ADR 0043 §*Restart surfaces* — the planned `checkpoint` / `from_checkpoint`
  first-class API this recipe anticipates.
- Ladruno fork [PR #577](https://github.com/nmorabowen/OpenSees/pull/577) — the
  element-`rho` serialization fix that makes transient restarts correct and
  deterministic (`LEDGER_quirks.md`: *element rho un-serialized*).
