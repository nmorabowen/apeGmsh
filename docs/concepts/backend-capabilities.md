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

When your own code talks to the model the bridge built, such as a custom
analysis loop after `ops.run()` or a query after `ops.analyze()`, take the
module from the same resolver rather than importing one by name:

```python
from apeGmsh.opensees.emitter.live import get_ops

opspy = get_ops()     # the module, and so the domain, the bridge drives
```

Beside a fork build, `import openseespy.opensees` binds a second module
with its own, empty domain, and every query answers from that one.

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
| **Gated by apeGmsh** | `RuntimeError` naming the fork, what still works, and the stock alternative | Elements, integrators, the `LadrunoProjection` handler, contact, FEAST, profiler, the modal family |
| **Rejected by the engine** | `OpenSeesError: See stderr output`, with an `unknown …` warning on stderr | Materials, `system Pardiso`, the fork recorders |

The first class exists because those commands are the ones stock OpenSees
would otherwise *accept*. The fork integrators, for example, are unknown to
stock, yet `integrator ExplicitBathe` is accepted silently and the analysis
steps on with the previous integrator — a converged wrong answer instead of
an error. The gates convert that into a refusal.

A few stock behaviours are refused for the same reason even though nothing
in them is fork-only. See [Stock engine defects](#stock-engine-defects).

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
`LadrunoGeneralizedAlpha`, `LadrunoHHT`, `LadrunoIndirectControl`,
`LadrunoLoadControl`
<!-- /capability-map:integrators -->

`LadrunoLoadControl` also needs a *recent* fork build (2026-08-04; its
default `-tangentPredictor` 2026-09-04). An older fork build accepts the
unknown integrator and keeps the previous one, so the in-process run checks
for the fork's `ladrunoLoadControl` command and then confirms the predictor
was armed — an older build is refused, not run as stock `LoadControl`.

Stock schemes — `Newmark`, `HHT`, `CentralDifference`,
`ExplicitDifference`, `LoadControl`, `DisplacementControl`, `ArcLength` —
are unaffected and run on any build.

### Constraints and coupling

- **`enforce="equation"` ties are *not* fork-only.** `g.constraints.tie(...)`
  and `Assembly.couple(...)` with `enforce="equation"` emit
  `equationConstraint` (EQ_Constraint, ADR 0068), which stock has had since
  OpenSees 3.8.0 (openseespy ≥ 3.8.0; on Linux that needs Python ≥ 3.12). On
  stock the rows are enforced exactly, and the in-process run works for
  **one tied model per process** — see
  [Stock engine defects](#stock-engine-defects). The same holds for
  **`ops.equation_constraint(...)`**, a hand-written row
  (`constrained=(node, dof)`, `retained=[(node, dof, coef), ...]`): `Lagrange`
  / `LadrunoProjection` auto-emitted, `Transformation` refused, serial only,
  and not archived by `ops.h5(...)`. What *is* fork-only is the explicit
  path: under an explicit integrator the bridge picks `LadrunoProjection`.
- **Contact.** `g.constraints.contact(...)` → `contactSurface` / `contact`, and
  `g.constraints.contact_plane(...)` → `contactPlane` (rigid analytical plane).
  Both lanes work in 2D as well as 3D, and both are **serial only** — parallel
  2D contact is out of scope in the fork and refused by name once the model is
  partitioned.
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

`LadrunoConcrete3D`'s `tension_law=` / `eps_fc=` / `gc_legacy=` need fork
build `1334d1e24` or later, and `flow_potential=` needs `916576661`. An older
build refuses the material (`unknown option`) rather than ignoring them. The
same `1334d1e24` commit changed the defaults (bilinear tension law, `Gc` read
as a compressive fracture energy), so a deck that sets none of them means
different things on either side of it; pin `tension_law` when that matters.

`LadrunoRCConcrete` / `LadrunoRCFiniteStrain`'s `beta_c=` and `cracked_nu=`
(`-betaC` / `-crackedNu`) are **not on any build of the fork's `ladruno`
branch yet**. Fork PR #873 was closed unmerged, and fork PR #877 re-lands
them. Unlike `LadrunoConcrete3D`, this parser ignores an unknown option, so a
build without them runs the deck with the flags discarded. The in-process run
therefore refuses them. `ops.tcl(...)` / `ops.py(...)` emit them with a
`LadrunoRCBuildWarning`. The floor is `LADRUNO_RC_C2_MIN_BUILD`, which stays
`None` until #877 merges. The `vc` tension-stiffening coefficient that
`tens_stiff_c=None` leaves to the build is **500** on the `ladruno` branch.
#877 changes it to 200, so pass `tens_stiff_c` explicitly to pin the curve.

The vanilla `DruckerPrager` runs on any build, but its tension-cutoff
return map is only correct from fork build `61b3efa04` on (fork ADR-95).
An older engine parses the same deck and answers differently — quadratic
solid elements hit a false collapse floor. Its two read-only diagnostics,
`material.ladrunoBranch` and `material.ladrunoTangent`, need that build
too: an empty `ops.eleResponse(e, "material", gp, "ladrunoBranch")` is
the probe for an older one.

`ASDPlasticMaterial3D` with a `DruckerPrager_YF` yield function has a
*second*, later floor — `67474aeb7` — for its apex classification (fork
ADR-94 wp/94f). The two floors are independent: `61b3efa04` fixes the UW
material's cutoff, `67474aeb7` fixes the ASD one's region test, and a
build can have the first without the second. Unlike the UW material, the
ASD one exposes no response token, so there is nothing to probe — compare
the build stamp instead.

#### The minimum useful build, and how to check it

| Want | Minimum fork build |
|---|---|
| A believable UW `DruckerPrager` tension cutoff, and the `ladrunoBranch` diagnostic | `61b3efa04` |
| A believable `ASDPlasticMaterial3D` + `DruckerPrager_YF` apex | `67474aeb7` |

Neither is enforced — a bare hash cannot prove ancestry, and an older
engine parses the identical deck. What it gets wrong is the answer, so
there is nothing to refuse at construction. Two ways to find out where
you stand:

```python
from apeGmsh.opensees._element_capabilities import probe_ladruno_branch
from apeGmsh.opensees.emitter.live import get_backend_build

get_backend_build()                      # the exact commit, or None off-fork
probe_ladruno_branch(ops_module, e_tag)  # False on a pre-61b3efa04 engine
```

The probe must be aimed at an element of a class that forwards
`material <gp> <token>` to its NDMaterial — `LadrunoBrick`,
`LadrunoBrick20`, `BezierTet10`, `TenNodeTetrahedron` — using a UW
`DruckerPrager`. Anywhere else it answers `False` for reasons that have
nothing to do with the build.

Once the response is there, `results.elements.gauss` reads it back under
eight names (`dp_branch`, `dp_gamma_cone`, `dp_gamma_cutoff`,
`dp_f1_trial`, `dp_f2_trial`, `dp_forced_accept`, `dp_i1`,
`dp_det_a_min`) and offers two censuses over it — `corner_census()` for
`dp_branch == 3`, and the material-agnostic `tension_census()` for
`mean_stress >= 0`. See
[ADR 0108](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0108-ladruno-branch-read-back.md).

### Recorders

`recorder ladruno` (the HDF5 `.ladruno` recorder) and `recorder Monitor`
(live SWMR telemetry). Every other recorder — including the plain text
recorders that [`Results.from_recorders`](results.md) reads — works on any
build.

Material-level response tokens (`material.<token>`) are a fork-recorder
feature: the fork splits them into `material <k> <token>` per Gauss point.
The bare spelling records nothing, so `ops.recorder.Ladruno` / `MPCO`
refuse it and name the prefixed form.

## Stock engine defects

Two stock OpenSees behaviours give a converged wrong answer with no warning,
so the in-process run refuses them on a stock build. Both are fixed on the
fork; deck emission is unaffected.

**`TenNodeTetrahedron` is 6× too soft.** Upstream's element applies the
tetrahedral 1/6 volume factor twice, so its stiffness, mass, body force and
reactions are all exactly 6× too small. That holds for every stock release
through openseespy 3.8.0, and for upstream master as of 2026-09-25. The
fork fixed it in PR #520. On stock, `ops.element.TenNodeTetrahedron` raises
in the live run; mesh tet4 (`FourNodeTetrahedron`) or hexahedra instead. A
fork build without the `ladrunoBuild` stamp (older than 2026-08-10) may
predate the fix, so it runs with a `Tet10UnverifiedBuildWarning`.

**`wipe()` keeps equation-tie rows.** Upstream `Domain::clearAll()` clears
nodes, elements, SP/MP constraints and patterns, but not `equationConstraint`
rows, and no stock command removes one. So the rows of one model survive
`wipe()` into the next, and the next model enforces them. In one measured
case the second model converged to twice the right stiffness; a model that
lacks a stale row's node aborts the process. The first tied model in a
process is exact. After it, the live emitter refuses to start another model
in that process on a stock build. Restart the process (or the kernel), or
run tied decks with `ops.tcl(run=True)` / `ops.py(run=True)`, which start
a fresh one. The fork clears the rows (fork PR #312).

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
| `partition-pymetis` | Weighted mesh partitioning (no Windows wheel) |
| `all` | `opensees` + `viewer` + `plot` + `dxf` + `mcp` |

`all` covers everything that installs cleanly everywhere, Studio
included:

```bash
pip install "apeGmsh[all]"
```

`scripts/make-venv.bat` builds a venv this way under `C:\venv\<name>`.

!!! note "What `all` leaves out, and why"

    `animation` (ffmpeg is a large binary payload) and the two
    `partition-*` extras. `pymetis` has no PyPI Windows wheel — it comes
    from conda-forge — so folding it into `all` would break
    `pip install "apeGmsh[all]"` on Windows; `partition-networkx` is
    inert without `nxmetis`, which installs only from git. Ask for those
    by name when you need them.

    Going the other way, `all` is broad: the Qt and VTK render stack, and
    through `mcp` a small server stack (`uvicorn`, `starlette`,
    `pydantic`, `opentelemetry-api`). Narrow it when you do not need all
    of that:

    ```bash
    pip install "apeGmsh[viewer,opensees]"
    ```

    Before 2026-08 `all` also omitted `mcp`, installing the Studio package
    without the SDK to start it. On an older release, add `"mcp>=1.2"`
    by hand.

## Related

- [The OpenSees bridge](opensees-bridge.md) — how primitives reach a deck.
- [Tie non-matching meshes](../how-to/tie-meshes.md) — choosing an
  `enforce` mode.
- [Drive Studio / MCP habitat](../how-to/studio-habitat.md).
