# Backend capabilities

Most of apeGmsh needs nothing but `pip install apeGmsh`. Solving needs an
OpenSees backend, and a minority of the OpenSees surface needs one
*particular* backend — the Ladruno fork. This page is the map: what runs
where, how a missing capability announces itself, and how to check the
environment you are actually in.

The short version:

| Tier | Needs | What it covers |
|---|---|---|
| **0** | nothing beyond the install extras | Geometry, meshing, parts, the FEM broker, `model.h5`, loads / masses / constraints, **deck emission**, results, viewers, sections, Studio |
| **1** | stock `openseespy` | In-process solving: `ops.run()`, `ops.analyze()`, `ops.eigen()`, domain capture |
| **2** | the **Ladruno fork** build | The primitives listed under [The fork-only surface](#the-fork-only-surface) |

!!! tip "Deck emission is never gated"

    `ops.tcl(...)` and `ops.py(...)` write a runnable deck for **every**
    primitive on **every** build, fork-only ones included. You can model
    and emit on a laptop with stock openseespy — or with no OpenSees at
    all — and run the deck on a machine that has the fork. Only the
    *in-process* run (`ops.run()` / `ops.analyze()`) is gated.

## Which build am I on?

```bash
python -m apeGmsh doctor
```

Finding `D5` names the resolved backend. A `warn D5` reading
*"Stock openseespy backend"* is normal and not an error — it means tier 2
is unavailable, nothing more.

In code:

```python
from apeGmsh.opensees import apeSees

ops = apeSees(fem)
ops.capabilities().has_fork      # True on a Ladruno build
```

Resolution order is `APEGMSH_OPENSEES_BIN` → a bare `import opensees` →
`import openseespy.opensees`. To use a fork build, point the environment
variable at the folder holding `opensees.pyd` **before** the first emit:

```bash
set APEGMSH_OPENSEES_BIN=C:\path\to\Ladruno\dist\bin
```

!!! warning "The fork is a source build"

    The Ladruno fork lives at
    [nmorabowen/OpenSees](https://github.com/nmorabowen/OpenSees) and
    publishes no wheels or binary releases. Tier 2 currently means
    compiling OpenSees yourself. If you are evaluating apeGmsh, plan on
    tiers 0 and 1 and treat tier 2 as opt-in.

## How a missing capability fails

Nothing in tier 2 fails *silently* — but the two classes read differently,
so it is worth knowing which you are looking at.

| Class | What you see | Which primitives |
|---|---|---|
| **Gated by apeGmsh** | `RuntimeError` naming the fork, what still works, and the stock alternative | Elements, integrators, equation ties, contact, FEAST, profiler, the modal family |
| **Rejected by the engine** | `OpenSeesError: See stderr output`, with an `unknown …` warning on stderr | Materials, `system Pardiso`, the fork recorders |

The first class exists because those commands are the ones stock OpenSees
would otherwise *accept* — `equationConstraint` and the fork integrators
are real symbols on a stock build, and an ungated call returns a converged
wrong answer instead of an error. The gates convert that into a refusal.

## The fork-only surface

### Elements

<!-- capability-map:elements -->
`BezierTet10`, `BezierTri6`, `LadrunoBrick`, `LadrunoBrick20`,
`LadrunoCST`, `LadrunoDispBeamColumn`, `LadrunoDistributingCoupling`,
`LadrunoEmbeddedNode`, `LadrunoEmbeddedRebar`, `LadrunoIMKBeam`,
`LadrunoKinematicCoupling`, `LadrunoLST`, `LadrunoQuad`,
`LadrunoRigidBody`, `LadrunoUP`
<!-- /capability-map:elements -->

Reached through `ops.element.*`, and also indirectly:
`g.reinforce` (embedded rebar), `g.embed` (embedded node), and
`g.constraints.kinematic_coupling` / `distributing_coupling` (RBE2 / RBE3)
all emit elements from this list.

### Integrators

<!-- capability-map:integrators -->
`CentralDifferenceLadruno`, `CentralDifferenceSMS`, `ExplicitBathe`,
`ExplicitBatheLNVD`, `ExplicitBatheLNVDSMS`, `ExplicitBatheSMS`,
`LadrunoArcLength`, `LadrunoDynamicRelaxation`,
`LadrunoGeneralizedAlpha`, `LadrunoHHT`, `LadrunoIndirectControl`
<!-- /capability-map:integrators -->

Stock schemes — `Newmark`, `HHT`, `CentralDifference`,
`ExplicitDifference`, `LoadControl`, `DisplacementControl`, `ArcLength` —
are unaffected and run on any build.

### Constraints and coupling

- **`enforce="equation"` ties.** `g.constraints.tie(...)` and
  `Assembly.couple(...)` with `enforce="equation"` emit
  `equationConstraint` (EQ_Constraint, ADR 0068). Fork-only **for the live
  run**. On stock, use `enforce="penalty"` with a tuned `stiffness` — see
  [Tie non-matching meshes](../how-to/tie-meshes.md).
- **Contact.** `g.constraints.contact(...)` → `contactSurface` / `contact`.
- **`LadrunoProjection`** constraint handler, and the
  `ladrunoProjectionTieForce` query.

### Analysis and solvers

- **`ops.eigen_feast(...)`** — band-targeted FEAST eigensolver.
- **`ops.complex_eigen(...)`** — complex / state-space modal.
- **Modal family** — `ops.modal_response_history(...)`,
  `responseSpectrumAnalysis` with `-combine`,
  `ops.frequency_response(...)`, `ops.steady_state_dynamics(...)`,
  `ops.random_response(...)`.
- **`ops.profiler` / `analyze(profile=...)`**, and
  `ops.critical_time_step()` / `ops.analyze_explicit(...)`.
- **`system Pardiso`** — threaded MKL sparse-direct.

### Materials

`LadrunoBondSlip`, `LadrunoCohesiveHinge`, `LadrunoCohesiveHingeBiaxial`,
`LadrunoConcrete3D`, `LadrunoJ2`, `LadrunoJ2Finite`, `LadrunoRCConcrete`,
`LadrunoRCFiniteStrain`, `LadrunoRebarBuckling`, `LadrunoUniaxialJ2`

### Recorders

`recorder ladruno` (the HDF5 `.ladruno` recorder) and `recorder Monitor`
(live SWMR telemetry). Every other recorder — including the plain text
recorders that [`Results.from_recorders`](results.md) reads — works on any
build.

## Install extras

Everything above assumes the right extras are installed. `apeGmsh` itself
pulls only `gmsh`, `h5py`, `numpy` and `pandas`.

| Extra | Enables |
|---|---|
| `opensees` | Stock `openseespy` — tier 1 |
| `viewer` | Qt + web viewers (`PySide6`, `pyvista`, `vtk`, `trame`) |
| `plot` | `matplotlib` / `scipy` plotting helpers |
| `dxf` | DXF import / export |
| `animation` | Video export from the results viewer |
| `mcp` | The Studio MCP server (`python -m apeGmsh.studio.mcp`) |
| `partition-pymetis` | Weighted mesh partitioning |
| `all` | Everything above **except `mcp`** |

!!! note "`all` does not include `mcp`"

    `pip install "apeGmsh[all]"` installs the Studio package but not the
    MCP SDK it needs to start. For the Studio habitat ask for it
    explicitly:

    ```bash
    pip install "apeGmsh[all]" "mcp>=1.2"
    ```

    `scripts/make-venv.bat` does this for you.

## Related

- [The OpenSees bridge](opensees-bridge.md) — how primitives reach a deck.
- [Tie non-matching meshes](../how-to/tie-meshes.md) — choosing an
  `enforce` mode.
- [Drive Studio / MCP habitat](../how-to/studio-habitat.md).
