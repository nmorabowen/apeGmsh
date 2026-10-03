# STKO translator — implementation contract and export rules

The build contract for [ADR 0111](../architecture/decisions/0111-stko-translator.md):
the module split, the exact interfaces, the shared types (verbatim), the type
registry, every STKO export rule the translator must reproduce (with the numbers
that verified it), and the deck-parity contract. Three implementers (mesh,
properties, conditions) and one integrator build against this file **without
talking to each other**. If something here is wrong or missing, stop and tell the
orchestrator; do not improvise an interface.

Written 2026-10-02 against `feat/interop-stko-translate` (apeGmsh `main` + the
reader of PR #1267). Units are whatever the document uses (San Ramon: N, mm,
tonne, s).

## 0. Reading order

1. ADR 0111 (decisions, 10 minutes).
2. §2 (API) and §3 (shared types): the parts everyone depends on.
3. §4 (registry) and §5 (rules): the parts you implement, by rule id.
4. Your work package: §6 mesh, §7 properties, §8 conditions, §9 integrator.
5. §10 tests, §11 what is not done.

Repo rules still apply: `AGENTS.md`, `architecture/agent-onboarding.md`,
`architecture/testing.md`, `python scripts/check_quirks.py`, the CHANGELOG
fragment workflow (`internal_docs/changelog_workflow.md`), and, for the bridge
option in §9, `.claude/skills/apegmsh-bridge-feature/SKILL.md`. Run Python 3.11 with
`PYTHONPATH=<worktree>/src` (the editable install is main's). Never
`import openseespy`; run decks with
`C:\Users\nmora\Github\OpenSees_Compile\piles-bin\fd87e396d\bin\OpenSees.exe`
(Tcl, subprocess, stdin from `DEVNULL`).

## 1. Ground truth and evidence

**STKO's exporter is the specification.** STKO ships its OpenSees writer as Python
source at `C:\Program Files\STKO\external_solvers\opensees\`. When a rule below is
unclear, read the file it cites there (read-only; do not copy it into apeGmsh — port
the behaviour, in our own code).

| Area | STKO source file |
|---|---|
| node + mass writing, mass map | `utils/write_node.py` (`fill_node_mass_map`, `__write_mass_and_node`, `write_node_partition`) |
| shell connectivity order | `element_properties/shell/shell_utils.py` (`getNodeString`) |
| ASDShellQ4 | `element_properties/shell/ASDShellQ4.py` |
| elasticBeamColumn, forceBeamColumn | `element_properties/beam_column_elements/{elasticBeamColumn,forceBeamColumn,internalBeamColumnElement}.py` |
| geomTransf | `element_properties/utils/geomTransf.py` |
| sections | `physical_properties/sections/{Elastic,ElasticMembranePlateSection,LayeredShell,Fiber,RectangularFiberSection}.py` |
| materials | `physical_properties/materials/nD/{ASDConcrete3D,PlateFromPlaneStress,PlateRebar}.py`, `materials/uniaxial/{ASDConcrete1D,Hysteretic}.py` |
| BeamSectionProperty | `physical_properties/special_purpose/BeamSectionProperty.py` |
| masses | `conditions/Mass/{FaceMass,AutoEdgeMass,NodeMass}.py` |
| loads | `conditions/Loads/Force/{FaceForce,EdgeForce,NodeForce}.py` |
| fix, rigidDiaphragm | `conditions/Constraints/sp/fix.py`, `conditions/Constraints/mp/rigidDiaphragm.py` |
| patterns | `analysis_steps/Patterns/addPattern/loadPattern.py` |
| IMPL-EX dTime utility | `utils/time_increment_utils.py` |

**Oracles** (research repo `C:\Users\nmora\Documents\gitAPE\Epistemic Uncertanty`,
read-only for you): `models/stko-rev0-tcl/Tier_1/1A/1A_TH_000.scd` with STKO's Tcl in
`1A/input files/` (4 partitions); `Tier_1/{1B,1C,1D}/{case}_TH_000.scd` with the
campaign decks in `Tier_1/{case}/campaign_sta2_rup1/` (2, 4 and 8 partitions; real
records inline; `MANIFEST.md` gives provenance and hashes).

**Prototypes** (scratch, not shipped; read them, do not import them):
`C:\Users\nmora\AppData\Local\Temp\claude\C--Users-nmora-Documents-gitAPE-Epistemic-Uncertanty\ad58411f-2653-4132-b66a-4425f3a4d4c2\scratchpad\runs\translator\`

- `design/proto/stko_rules.py` — every rule of §5 as a pure function (≈300 lines).
- `design/proto/stko_deck.py` — a canonicaliser for STKO's partitioned Tcl (§9.3).
- `design/proto/verify.py 1A 1B 1C 1D` — rules vs decks: **132/132 checks pass**
  (outputs `design/proto/verify_<case>.txt`).
- `spike/spike_mesh_route.py`, `spike/spike_full.py <case> [--fem-tags]` — the mesh
  route of §6 at full scale, emitting through apeSees and comparing with STKO.

## 2. Pipeline and public API

```python
from apeGmsh import apeGmsh
from apeGmsh.interop.stko import read_scd, translate_scd, build_opensees

scd = read_scd("1C_TH_000.scd")
with apeGmsh(model_name="1C") as g:
    result = translate_scd(g, scd)                 # 1-4 below; raises UnsupportedSTKOTypes
    fem = g.mesh.queries.get_fem_data(dim=None)    # NEVER generate() or renumber()
ops = build_opensees(fem, result)                  # 5-7 below
ops.tcl("1C.tcl")
```

`translate_scd(g, scd, *, records=None)`:
1. `collect_unsupported(scd)` → raise `UnsupportedSTKOTypes(items)` if non-empty.
2. `translate_conditions.plan_conditions(scd, records=records) -> ConditionsPlan`
   (pure numpy, no session, no bridge; before the session, so a bad `records=`
   raises on an empty session).
3. `translate_mesh.build_mesh(g, scd) -> MeshMap` (session: entities, nodes,
   elements, PGs, diaphragm carriers built from `diaphragm_groups(scd)`, the same
   groups the plan's pairs come from).
4. `translate_conditions.declare_session(g, scd, plan, mesh)` (rigid diaphragms on
   `g.constraints`, STKO's links verbatim, C7).

Returns `TranslateResult(scd, mesh, plan, ignored)`.

`build_opensees(fem, result, *, element_tags="fem", stages=True) -> apeSees`:
5. `ops = apeSees(fem, element_tags=element_tags)` (§9.1; a bridge without the
   option raises `TypeError`, never a silent renumbering); `ops.model(ndm=3, ndf=6)`.
6. `translate_props.build_props(ops, result.scd, result.mesh.element_groups) -> PropsResult`.
7. `translate_conditions.build_conditions(ops, result.plan, result.mesh, stages=stages)`.

Nothing is exported by the translator itself; the caller picks `ops.tcl` / `ops.py` /
`ops.h5`. `build_props` and `build_conditions` are public so a recipe can use one
without the other (ADR 0072's lesson).

## 3. Shared types — `src/apeGmsh/interop/stko/translate_types.py` (verbatim)

Whoever starts first creates this file **exactly** as below; nobody edits it. A
change goes through the orchestrator and this section first.

```python
"""Types and helpers the STKO translator modules exchange (ADR 0111).

Fixed by internal_docs/stko_translator_rules.md section 3: change it there first.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .model import ScdModel

Vec3 = tuple[float, float, float]
Vec6 = tuple[float, float, float, float, float, float]
#: (geometry id, kind, 0-based sub-shape index), kind in KIND_DIM.
SubShape = tuple[int, str, int]

KIND_DIM: dict[str, int] = {"vertices": 0, "edges": 1, "faces": 2, "solids": 3}
#: SubShapeRef.kind (interaction masters / slaves) -> kind.
REF_KIND: dict[int, str] = {1: "vertices", 2: "edges", 3: "faces", 4: "solids"}

Category = Literal[
    "element_property", "physical_property", "condition", "definition",
    "analysis_step", "interaction", "mesh_element", "option",
]


@dataclass(frozen=True, slots=True)
class Unsupported:
    """One STKO object, or one option of an object, the translator will not translate."""

    category: Category
    xobj_meta: str              # STKO type; "<type>:<attribute>" for an option
    ids: tuple[int, ...]
    names: tuple[str, ...]
    reason: str                 # "tier-only: not in this version" | "unknown STKO type" | option text


class UnsupportedSTKOTypes(NotImplementedError):
    """Raised before the session is touched; lists every offender."""

    def __init__(self, items: tuple[Unsupported, ...]) -> None:
        self.items = items
        lines = [
            f"  [{u.category}] {u.xobj_meta} ids={list(u.ids)} names={list(u.names)}: {u.reason}"
            for u in items
        ]
        super().__init__(
            f"{len(items)} STKO type(s) or option(s) are not supported by this version "
            "of the translator:\n" + "\n".join(lines)
        )


@dataclass(frozen=True, slots=True)
class Ignored:
    """An STKO object deliberately not translated (it is not part of the model)."""

    category: Category
    xobj_meta: str
    ids: tuple[int, ...]
    names: tuple[str, ...]
    reason: str
    payload: str | None = None  # e.g. the Tcl of a customCommand


@dataclass(frozen=True, slots=True)
class ElementGroup:
    """Analysis elements sharing (element property, physical property, local axis)."""

    pg: str
    dim: int                    # 1 beams, 2 shells
    element_property: int
    physical_property: int
    local_axis: Vec3            # shell: -local x vector; beam: geomTransf vecxz (local_axis())
    element_ids: tuple[int, ...]  # STKO ids, ascending


@dataclass(frozen=True, slots=True)
class DiaphragmGroup:
    """One rigid-diaphragm master and its slaves (from the interaction's links)."""

    condition: int
    perp_dirn: int              # 1, 2 or 3
    master_node: int
    slave_nodes: tuple[int, ...]  # in link order
    master_pg: str
    slave_pg: str


@dataclass(frozen=True, slots=True)
class MeshMap:
    node_ids: tuple[int, ...]                                # session nodes, ascending
    element_groups: tuple[ElementGroup, ...]
    subshape_entities: Mapping[SubShape, tuple[int, ...]]   # gmsh tags, dim KIND_DIM[kind]
    selection_set_pgs: Mapping[str, Mapping[str, str]]      # set -> kind -> PG
    condition_pgs: Mapping[int, Mapping[str, str]]           # condition id -> kind -> PG
    diaphragms: tuple[DiaphragmGroup, ...]
    carrier_ids: range                                       # synthetic element ids, never emitted


@dataclass(frozen=True, slots=True)
class PropsResult:
    primitives: Mapping[int, Any]   # STKO physical-property id -> apeSees primitive
    elements: Mapping[str, Any]     # ElementGroup.pg -> element primitive
    transforms: Mapping[Vec3, Any]  # vecxz -> geomTransf primitive


@dataclass(frozen=True, slots=True)
class TimeSeriesSpec:
    stko_id: int
    type: Literal["Linear", "Path"]
    factor: float = 1.0
    dt: float | None = None
    values: tuple[float, ...] | None = None
    start_time: float | None = None   # None: not written
    placeholder: bool = False         # values are the .scd's own, no record supplied


@dataclass(frozen=True, slots=True)
class PatternSpec:
    stko_id: int                      # STKO analysis-step id (STKO's pattern tag)
    name: str
    kind: Literal["Plain", "UniformExcitation"]
    series: int                       # TimeSeriesSpec.stko_id
    loads: Mapping[int, Vec6] = field(default_factory=dict)  # node -> summed load
    conditions: tuple[int, ...] = ()  # load + massToLoad condition ids, STKO order
    direction: int | None = None      # UniformExcitation: 1..6


@dataclass(frozen=True, slots=True)
class RayleighSpec:
    alpha_m: float
    beta_k: float
    beta_k_init: float
    beta_k_comm: float


@dataclass(frozen=True, slots=True)
class StageSpec:
    stko_id: int
    name: str
    analysis: Literal["Static", "Transient"]
    patterns: tuple[int, ...]         # PatternSpec.stko_id first active in this stage
    n_incr: int
    duration: float
    load_const: bool
    integrator: tuple[str, tuple[float, ...]]  # ("LoadControl", (dlam,)), ("Newmark", (gamma, beta))
    chain: Mapping[str, Any]          # STKO's solver choices, informational


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """A condition as intent: the definitions source for the parametric recipes."""

    stko_id: int
    name: str
    type: str
    value: tuple[float, ...]          # per area / length / node; rho for AutoEdgeMass
    per: Literal["area", "length", "node", "volume", "none"]
    pgs: Mapping[str, str]            # kind -> PG (MeshMap.condition_pgs)
    patterns: tuple[int, ...]         # load patterns using it
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConditionsPlan:
    masses: Mapping[int, Vec6]
    fixes: Mapping[tuple[int, ...], tuple[int, ...]]   # 6-DOF mask -> nodes, ascending
    diaphragm_pairs: frozenset[tuple[int, int, int]]   # (perp, master, slave)
    time_series: tuple[TimeSeriesSpec, ...]
    patterns: tuple[PatternSpec, ...]                  # STKO step order
    constraint_conditions: tuple[int, ...]              # sp + mp of the constraint pattern
    rayleigh: RayleighSpec | None
    stages: tuple[StageSpec, ...]
    implex_dt_targets: tuple[int, ...]
    summaries: tuple[ConditionSummary, ...]
    ignored: tuple[Ignored, ...]


@dataclass(frozen=True, slots=True)
class TranslateResult:
    scd: "ScdModel"
    mesh: MeshMap
    plan: ConditionsPlan
    ignored: tuple[Ignored, ...]


def quat_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of STKO's element quaternion ``(x, y, z, w)``;
    columns are the element's local x, y, z axes."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def local_axis(scd: "ScdModel", eid: int) -> Vec3:
    """Rules M4/M5: a shell's ``-local`` vector (column 0, STKO prints %.6g) or a
    beam's ``vecxz`` (column 2, rounded to 12 significant digits for grouping)."""
    M = quat_matrix(scd.mesh.orientation[eid])
    if len(scd.mesh.elements[eid].nodes) == 2:
        return tuple(float(f"{v:.12g}") + 0.0 for v in M[:, 2])  # type: ignore[return-value]
    return tuple(float(f"{v:.6g}") + 0.0 for v in M[:, 0])       # type: ignore[return-value]


def property_references(scd: "ScdModel", pid: int) -> frozenset[int]:
    """Physical properties ``pid`` references, transitively: its INDEX / INDEX_VEC
    attributes, plus the fiber-group materials inside a fiber section's custom
    object (which are not INDEX attributes)."""
    seen: set[int] = set()
    stack = [pid]
    while stack:
        x = scd.physical_properties[stack.pop()]
        refs: list[int] = []
        for name in x.references:
            v = x.attributes[name]
            refs.extend(v if isinstance(v, tuple) else (v,))
        fs = x.attributes.get("Fiber section")
        if isinstance(fs, dict):
            for grp in ("PUNCTUAL_FIBER_GROUPS", "SURFACE_FIBER_GROUPS", "LINEAR_FIBER_GROUPS"):
                for item in (fs.get(grp) or {}).values():
                    refs.append(int(np.asarray(item["@attrs"]["PHYS_PROP_ID"]).ravel()[0]))
        for r in refs:
            if r and r in scd.physical_properties and r not in seen:
                seen.add(r)
                stack.append(r)
    return frozenset(seen)
```

## 4. The type registry

The registry is data the three modules each own a slice of (each module's
`check_supported(scd) -> list[Unsupported]` and its handlers). Status: **S**
supported, **T** tier-only (error "tier-only: not in this version"), **I** ignored
(returned in `ignored`, not an error). Anything **not listed** in a module's slice is
an error "unknown STKO type". The table lists every type in the 16 San Ramon
documents (survey of 2026-10-02).

| Category | `XOBJ_META` | Status | Owner | Cases |
|---|---|---|---|---|
| element property | `shell.ASDShellQ4` | S | props | all |
| element property | `beam_column_elements.elasticBeamColumn` | S | props | 1A 2A 3A 4A |
| element property | `beam_column_elements.forceBeamColumn` | S | props | B, C, D cases |
| element property | `brick_elements.stdBrick` | T | props | 3x 4x |
| element property | `zero_length_elements.zeroLength` | T | props | 2x |
| element property | `absorbingBoundaries.ASDAbsorbingBoundary3DAuto` | T | props | 3x |
| physical property | `sections.Elastic` | S | props | 1A 2A 3A 4A |
| physical property | `sections.ElasticMembranePlateSection` | S | props | all |
| physical property | `sections.LayeredShell` | S | props | B, C, D |
| physical property | `sections.Fiber` | S | props | 1B 2B 3B 4B |
| physical property | `sections.RectangularFiberSection` | S | props | B, C, D |
| physical property | `special_purpose.BeamSectionProperty` | S | props | B, C, D |
| physical property | `materials.nD.ASDConcrete3D` | S | props | B, C, D |
| physical property | `materials.nD.PlateFromPlaneStress` | S | props | B, C, D |
| physical property | `materials.nD.PlateRebar` | S | props | B, C, D |
| physical property | `materials.uniaxial.ASDConcrete1D` | S | props | B, C, D |
| physical property | `materials.uniaxial.Hysteretic` | S | props | B, C, D |
| physical property | `materials.nD.ElasticIsotropic` | T | props | 3x 4x |
| physical property | `materials.nD.ASDAbsorbingBoundary3DMaterial` | T | props | 3x |
| physical property | `special_purpose.zeroLengthMaterial`, `materials.uniaxial.Elastic` | T | props | 2x |
| condition | `Constraints.sp.fix` | S | conditions | all |
| condition | `Constraints.mp.rigidDiaphragm` | S | conditions | 1B 1C 2B 2C 3B 3C 4B 4C |
| condition | `Mass.FaceMass`, `Mass.AutoEdgeMass`, `Mass.NodeMass` | S | conditions | all / all / B, C |
| condition | `Loads.Force.FaceForce`, `Loads.Force.EdgeForce`, `Loads.Force.NodeForce` | S | conditions | all / B, C / B, C |
| condition | `Constraints.mp.ASDEmbeddedNodeElement` | T | conditions | 3x 4x |
| condition | `Loads.Generic.H5DRM` | T | conditions | 4x |
| definition | `timeSeries.Linear`, `timeSeries.Path` | S | conditions | all |
| analysis step | `Patterns.addPattern.loadPattern`, `.constraintPattern`, `.UniformExcitation` | S | conditions | all (UE: tiers 1-3) |
| analysis step | `Misc_commands.rayleigh`, `Analyses.AnalysesCommand` | S | conditions | all |
| analysis step | `Patterns.addPattern.H5DRM`, `Misc_commands.ASDAbsorbingBoundaryActivate` | T | conditions | 4x / 3x |
| analysis step | `Recorders.MPCORecorder`, `Misc_commands.region`, `Misc_commands.monitor` | I | conditions | all |
| analysis step | `Misc_commands.customCommand` | I (payload = Tcl) | conditions | all |
| analysis step | `Misc_commands.ImplexAutoErrorControlActivate` | I only after the last analysis, else option-unsupported | conditions | B, C, D |
| interaction | `NN` whose links carry `Constraints.mp.rigidDiaphragm` | S | mesh | 1B 1C … |
| interaction | `NN` used by anything else (2A links), `NE` (embedded) | T | mesh | 2x, 3x 4x |
| mesh element | 102 `line2`, 303 `quad4`, 600 `link` (diaphragm only) | S | mesh | — |
| mesh element | 401 / 500 `hex8` | T | mesh | 3x 4x |

**What is scanned** (`translate.collect_unsupported`, the union of the three
`check_supported`): element properties of analysis elements and the element and
physical properties every interaction carries (props slice: 2A's `zeroLength`,
`zeroLengthMaterial` and `uniaxial.Elastic` are named, and a *supported* element
type on an interaction is refused as `<type>:on interaction`); the
physical-property closure of those (`property_references`); every `Mass.*`
condition (STKO applies all masses whatever the patterns say); conditions referenced
by any loadPattern / constraintPattern; every definition referenced by a pattern;
every analysis step; every interaction; every mesh element type of analysis
elements. Unreferenced templates are not scanned (1B defines `zeroLengthMaterial`
and `materials.uniaxial.Elastic` and assigns neither: it must translate).

**Option-level unsupported** (category `option`, `xobj_meta="<type>:<attribute>"`),
checked on scanned objects only — each owner lists its own:

- props: ASDShellQ4 nothing; elasticBeamColumn `-releasey`/`-releasez`, non-zero
  `Y/section_offset`/`Z/section_offset` on its section (STKO writes `-jntOffset`),
  `-cMass`; sections.Elastic with `Section/PROPS[1] != PROPS[2]` (P2: the Iyy/Izz
  order is unverified) or fewer than 4 `PROPS`; forceBeamColumn `-cMass`, BeamSectionProperty `Option` other than
  `StandardIntegrationTypes`; sections.Elastic `Use Uniaxial Materials` (STKO writes an
  Aggregator); Fiber/RectangularFiberSection `-torsion`, RectangularFiberSection with no
  core material (STKO generates a confined material on the fly); ASDConcrete3D/1D
  `Preset` other than `Concrete (9P)`, `implexAlpha != 1.0`, `-crackPlanes`,
  `constitutiveTensorType = Tangent` with IMPL-EX; geomTransf `transfType` other than
  Linear / PDelta / Corotational.
- conditions: any `Mode = "function"` on a scanned condition (1C's unreferenced
  `loads_Lateral_X` has `Fx = 1.0908622118874465e-11*z**1.5`: it is not scanned, so 1C
  translates); FaceForce/EdgeForce `Global = False` is supported by rule C3 but
  unexercised; AutoEdgeMass `Type = "eleLoad"`; loadPattern `-fact`, non-empty `eleLoad`,
  `sp` or `genericLoad`; UniformExcitation `-fact`, `-vel0`, `-disp`, `-vel`; a node
  fixed by two sp conditions with different masks; a node slave in two diaphragms,
  a node in two diaphragm groups (as master or slave), a master listed as its own
  slave, a slave at its master's coordinates (C7); `ImplexAutoErrorControlActivate`
  with an Analyses command after it; a static Analyses command after a transient
  one; a test with `use_nType` on a static stage.
- mesh: a sub-shape whose analysis elements need more than one entity *and* a
  condition PG on it (never in Tier 1; handled by the split, listed only if the split
  cannot represent it). **Node-subset guard:** the session's nodes
  (`translate_mesh.session_nodes`: carriers, referenced vertices, every node of every
  element on a referenced edge/face/solid) must all be nodes of analysis elements
  (links excluded) or of referenced diaphragm groups — else
  `mesh:session nodes outside the analysis model` (they would be free nodes in our
  deck); and every mesh node must be in the session — else
  `mesh:mesh nodes outside the session` (STKO writes every mesh node:
  `write_node.write_node_not_assigned` writes the unused ones with `ndf 3`; an
  unreferenced diaphragm's links or an unassigned meshed geometry would be missing
  from ours). Tier 1: session = mesh = analysis nodes (14003 / 2547 / 3979 / 13459).

## 5. Export rules

Each rule: what STKO writes, the STKO source, and its evidence. **V** = verified
numerically on the Tier-1 exports (prototype `verify.py`); **H** = from STKO's source,
not exercised by Tier 1 (keep it, test it synthetically, verify when a case has it).

### Mesh and elements (owner: mesh, except M5's transform which props builds)

- **M1 Analysis elements.** OpenSees receives `ScdModel.analysis_elements()` minus
  interaction links: 1A 13704 (12968 ASDShellQ4 + 736 elasticBeamColumn), 1B 2456,
  1C 3704, 1D 13160 — equal to the deck's element sets, same ids. Link elements (type
  600, 360 in 1B, 840 in 1C) become `rigidDiaphragm` (C7), never elements. **V**
- **M2 Nodes.** The node set is every node of an analysis element including links
  (1A 14003, 1B 2547, 1C 3979, 1D 13459): identical to the deck. Coordinates are the
  `.scd`'s; STKO prints them rounded (max |Δ| 4.3e-6 mm on 4.1e4 mm). **V**
- **M3 Shell node order** (`shell_utils.getNodeString`). For a 4-node shell with
  nodes p0..p3 and local x `ux` (M4): `vx = normalize(p1 + p2 − p3 − p0)`; keep
  `(n0, n1, n2, n3)` if `|ux·vx| > 0.5`, else write `(n1, n2, n3, n0)`. Rotates 11134
  of 12968 shells in 1A, 1508/2408 in 1B, 1838/3512 in 1C; with it every element's
  connectivity equals the deck's, node order included. The mesh module adds the
  shell to gmsh in this order. **V**
- **M4 Shell local axes.** `-local` is column 0 of the element quaternion's rotation
  matrix (`quat_matrix`, `MESH/ELEMENT_ORIENTATION_QUATERNIONS`, stored `(x, y, z,
  w)`), printed `%.6g`: 0 mismatches on 12968 (1A) / 2408 / 3512 / 12968 shells. The
  rule "x-normal walls take `0 -1 0`" in `FINDINGS.md` is a consequence, not the rule.
  The quaternion already includes assigned local axes. **V**
- **M5 Beam `vecxz`.** `geomTransf <transfType> <eleTag> vZ` with `vZ` = column 2 of
  the same matrix, full precision, transform tag = element tag; 0 mismatches (736
  elastic in 1A; 48 W6500x35 wall-beams in 1B, which carry the `rotatedWalls` local
  axes; 192 columns in 1C/1D). A non-zero section offset adds `-jntOffset` (option,
  unsupported). **V**
- **M6 Grouping.** On Tier 1 every analysis sub-shape has exactly one local axis
  (0 sub-shapes with two), giving 7 / 5 / 8 / 9 element groups (1A/1B/1C/1D). **V**

### Properties (owner: props)

STKO writes each physical property once, with tag = its id, and writes **every**
defined property including unreachable templates (1B exports 26 of 30, of which 18
unreachable). Translate the closure reachable from element groups only.

- **P1 ASDShellQ4** (`ASDShellQ4.py`): `element ASDShellQ4 tag n1..n4 secTag
  [-corotational if Kinematics == "Corotational"] [-noeas if not Use EAS]
  [-drillingStab %.6g(Drilling Stabilization) if Drilling DOF Type == "Elastic Drilling
  DOF" else -drillingNL] -local x y z`. secTag = the face's physical property id
  (12968/12968). apeSees: `ops.element.ASDShellQ4(pg, section, corotational, no_eas,
  drilling_stab | drilling_nl=True, local_cs=group.local_axis)`. Flags follow each
  element property: 1A `-drillingStab 0.01` everywhere; 1B basement walls `-noeas
  -drillingStab 0.01` and foundation `-noeas -drillingNL`; 1C/1D `-noeas
  -drillingNL`. **V**
- **P2 elasticBeamColumn** (`elasticBeamColumn.py`): arguments `A E G J Iy Iz transf`
  with `A = PROPS[0]·A_modifier`, `Iy = PROPS[1]·Iyy_modifier`, `Iz =
  PROPS[2]·Izz_modifier`, `J = PROPS[3]·J_modifier` (`PROPS` = the section's
  `Section/PROPS` array), `E`, `G` from the sections.Elastic attributes; `-mass
  massDens` if `-mass`. Relative error 0 on 736 elements. The section itself is
  **not** referenced by the element (STKO still writes `section Elastic` for it). **V**
  for square sections only. STKO's exporter reads `Section.properties.Iyy` / `.Izz`
  of the compiled `MpcBeamSection`, not `PROPS`; `PROPS` has 8 values, and
  `RectangularFiberSectionDomain.py` prints the same object's properties in the order
  `area, Iyy, Izz, J, alphaY, alphaZ, centroidY, centroidZ`, which *suggests*
  `PROPS[1] = Iyy`, `PROPS[2] = Izz` but does not prove the storage order. Every
  `sections.Elastic` in the 281 readable San Ramon documents is square (700×700,
  300×300; `PARAMS` = width, height), so the order cannot be told from data:
  `PROPS[1] != PROPS[2]` is refused (option `sections.Elastic:Section`) until a
  non-square case decides it. **H** (the Iy/Iz assignment)
- **P3 forceBeamColumn + BeamSectionProperty** (`internalBeamColumnElement.py`):
  `element forceBeamColumn tag i j transfTag <IntegrationType/1> <secTag/1>
  <numIntPts/1>` — the old inline integration form — when the BeamSectionProperty's
  `Option == "StandardIntegrationTypes"`; `-iter maxIters tol` if `-iter`, `-mass` if
  `-mass`. Tier 1: `Lobatto 11 5` (C70x70 columns), `Lobatto 24 2` (W6500x35). apeSees:
  `ops.beamIntegration.<Type>(section=prim[secTag/1], n_ip=numIntPts/1)` and
  `ops.element.forceBeamColumn(pg, transf, integration)`. 48/48 and 192/192. **V**
- **P4 geomTransf**: `transfType` of the element property (`Linear` in Tier 1);
  `ops.geomTransf.Linear(vecxz=group.local_axis)` once per distinct vecxz. **V**
- **P5 sections.ElasticMembranePlateSection**: `E nu h rho Ep_mod` from the
  attributes of the same names. **V**
- **P6 sections.LayeredShell** (`LayeredShell.py`): `n` then `(matTag_i,
  thickness_i)` from the `matTag` (INDEX_VEC) and `thickness` attributes, in order.
  A layer may reference a PlateFromPlaneStress, a PlateRebar **or a 3-D
  ASDConcrete3D directly** (1D's `Slab_NLShell` 19 uses material 6 in four layers;
  `ShellLayer` accepts a 3-D nDMaterial). Values exact for 17/18/19/20. **V**
- **P7 PlateFromPlaneStress / PlateRebar**: `matTag OutofPlaneModulus` →
  `PlateFromPlaneStress(material, G_out)`; `matTag sita` → `PlateRebar(material,
  angle)`. **V**
- **P8 Fiber sections** (`Fiber.py`, `RectangularFiberSection.py`): `section Fiber
  tag [-GJ GJ if -GJ] { fiber y z A mat … }`. The fibers are **stored** in the
  `Fiber section` custom object: each `{PUNCTUAL,SURFACE,LINEAR}_FIBER_GROUPS/ITEM_k`
  has `FIBERS` rows `(x, y, area)` and `@attrs/PHYS_PROP_ID` (the material). STKO
  writes punctual groups, then surface, then linear, items in `k` order, each fiber
  at `(x − cx, y − cy)` with `(cx, cy) = CENTER_AND_AREA[0:2]` (unless `-noCentroid`).
  Never recompute the fibers from Width/Height/Cover. Exact (|Δ| = 0) for 1B's
  W6500x35 (273 fibers, centroid at (21999.95, 25424.99) — the section is drawn in
  global coordinates) and C70x70 (276). apeSees: `ops.section.Fiber(fibers=tuple(
  FiberPoint(material, y, z, area)), GJ=...)`. **V**
- **P9 ASDConcrete3D / ASDConcrete1D, preset "Concrete (9P)"** (`ASDConcrete3D.py`;
  `ASDConcrete1D.py` has byte-identical law functions). Port these functions
  verbatim in behaviour (`stko_rules.py` has them: `lch_ref`, `tension`,
  `compression`, `asdconcrete_9p`):
  - inputs `E, fcp, fc0, fcr, ecp, ft, Gt, Gc, PScale Tension, PScale Compression`;
  - `lch_ref = min(hmin_t, hmin_c)`, `hmin_t = Gt / (ft²/(2E)) / 100`,
    `hmin_c = Gc / (fc·(ec − ec_pl)/2) / 100`, `ec_pl = 0.4·(ec − fc/E) + fc/E`;
  - tension `tension(E, ft, Gt/lch_ref, pscale_t)` (6 points: `0.9 ft` at `0.9ft/E`,
    `ft` at `1.5 ft/E`, then `0.2 ft`, `1e-3 ft`, plateau; damage from target plastic
    strains), compression `compression(E, fc, fc0, fcr, ec, Gc/lch_ref, pscale_c)` (13
    points: linear to `fc0`, 9 quadratic-Bézier points to `(ec, fc)`, linear softening
    to `fcr`, plateau);
  - written as `nDMaterial ASDConcrete3D tag E v -Te … -Ts … -Td … -Ce … -Cs … -Cd …
    -rho rho -eta eta -Kc Kc -cdf CDF [-implex -implexAlpha a if integration ==
    "IMPL-EX"] [-crackPlanes …] [-tangent] -autoRegularization lch_ref`; the 1-D form
    has no `v -rho -Kc -cdf`.
  Backbones relative error 0, `lch_ref` exact: 6.377551020408164 (UC), 4.081632653061225
  (CC) in 1B/1C; 3.188775510204082 (3-D, 1D). apeSees: construct the raw
  `ASDConcrete3D(E, v, Te…Cd, lch_ref, rho, Kc, eta, cdf, implex)` /
  `ASDConcrete1D(...)` classes and `ops.register(...)` them (the namespace methods
  build a different law from `fc`). apeSees omits `-implexAlpha`; OpenSees' default is
  1.0, so `implexAlpha != 1.0` is an unsupported option. **V**
- **P10 Hysteretic** (`Hysteretic.py`): `s1p e1p s2p e2p [s3p e3p] s1n e1n s2n e2n
  [s3n e3n] pinchx pinchy damage1 damage2 [beta if use_beta]`; exact. **V**
- **P11 Tags.** STKO: element tag = mesh id, section/material tag = physical-property
  id, geomTransf tag = element tag. Ours: element tag = STKO id with §9.1, everything
  else allocated by apeSees and compared structurally (§9.3).

### Conditions, patterns, steps (owner: conditions)

- **C1 Nodal mass** (`write_node.fill_node_mass_map`): the sum over **every**
  `Mass.*` condition in the document (no pattern involved), translational only
  (`-mass mx my mz 0 0 0`), written on the node; on a partitioned export only the
  copy on the node's own partition carries `-mass`.
  - FaceMass (`Mode = constant`, `mass` = 3-vector per area): per face element
    `∫ Nᵢ · m dA` (consistent; 2×2 Gauss on quad4 is exact for a constant).
  - AutoEdgeMass: per edge element `∫ Nᵢ · (ρ·A) dL` → `ρ·A·L/2` per end, with `A` =
    **mean** over the edge's physical property and the properties it references of:
    sections.Elastic → `Section/PROPS[0]` (no `A_modifier`), sections.Fiber → the sum of
    its **surface** fibers' areas (not `CENTER_AND_AREA[2]`, which includes rebar),
    sections.RectangularFiberSection → `Width·Height`. (1B's W6500x35: 2,275,000 mm²;
    using `CENTER_AND_AREA` gave +4.85 t, the 12,330.75 mm² of rebar.)
  - NodeMass: the `mass` 3-vector on each assigned vertex's node.
  Max |Δm| 5.0e-10 / 5.4e-10 / 5.0e-10 / 5.0e-10 t; totals equal to the last printed
  digit (11720.837250 t for 1A/1C/1D, 11736.737000 t for 1B). **V**
  Whether STKO's reference walk for the area is direct or transitive cannot be told
  on Tier 1 (one section per closure): use `property_references` and flag a closure
  with two sections as an unsupported option until a case decides it.
- **C2 FaceForce** (`FaceForce.py`): per face element, `F` (or the quaternion-rotated
  `R·F` if `Global = False`, **H**) lumped as `∫ Nᵢ F dA`, written as one `load` line
  per element corner — so sum per node: 1A pattern 9 has 53344 lines on 14003 nodes.
  Each element's lines are written on its own partition only, so summing all lines is
  exact. Moments 0. **V**
- **C3 EdgeForce** (`EdgeForce.py`): the same on edge elements, `∫ Nᵢ F dL`. **V** (1B/1C)
- **C4 NodeForce** (`NodeForce.py`): `F` on each assigned vertex's node, global, no
  moments, written once on the node's partition. **V** (1B/1C)
- **C5 AutoEdgeMass "Convert to load"** (`AutoEdgeMass.convertToLoad`): when a
  loadPattern lists the condition in `massToLoad` and `Convert to load` is true and
  `Type == "load"`: per edge element `∫ Nᵢ (g·ρ·A) dL` with `g = (gx, gy, gz)` and the
  same `A` as C1. San Ramon: `gz = −9810` (FaceForce self-weight used 9807: both are
  just values). **V**
- **C6 fix** (`fix.py`): nodes = every node of every mesh element (analysis or not)
  on the assigned edges/faces/solids, plus the assigned vertices' nodes; mask
  `(Ux, Uy, Uz, Rx, Ry, Rz/3D)` for a 6-dof node. Exact: 1551 (1A, 1D), 1563 = 1551 +
  12 masters with `0 0 1 1 1 0` (1B), 212 (1C). The condition PGs give the same node
  set (`fem.nodes.select(pg=...)`), checked. **V**
- **C7 rigidDiaphragm** (`rigidDiaphragm.py`): for each link element of each
  interaction of the condition, `rigidDiaphragm perpDirn nodes[0] nodes[1:]` (one
  command per link; repeated on every partition holding a slave). Canonical form is
  the set of `(perp, master, slave)`: 360 (1B) and 840 (1C) pairs, 12 masters, exact.
  The masters are separate one-vertex geometries with no element; their vertical
  translation and rocking are fixed by the `spDiaphragm` fix (C6). **V**
  Only conditions a constraint pattern lists under `mp` are written
  (`diaphragm_groups(scd)`: one `DiaphragmGroup` per (condition, master), slaves in
  link order). **Carrying them verbatim.** apeGmsh has no node-level master/slave
  diaphragm API; `g.constraints.rigid_diaphragm` resolves geometrically
  (`resolve_rigid_diaphragm`: nodes of `master_pg ∪ slave_pg` within
  `plane_tolerance` of the plane through `master_point`; master = the node nearest
  `master_point`, first over the whole model, then over those nodes). So: the two
  carriers own exactly `{master}` and the slaves; `master_point` = the master's own
  coordinates; `plane_tolerance = 2·max |slave offset from the master's plane| +
  1e-6·max(1, bbox diagonal)`, so no slave is dropped (OpenSees' `rigidDiaphragm`
  keeps an off-plane slave with a warning, `RigidDiaphragm.cpp`; the earlier fixed
  1e-6 tolerance dropped it). A slave at the master's coordinates, a node in two
  groups and a master that is its own slave are refused (§4). After `get_fem_data`
  `verify_diaphragms(fem, plan)` asserts the resolved records
  (`fem.nodes.constraints.rigid_diaphragms()`) equal `plan.diaphragm_pairs`, no pair
  twice; `build_conditions` runs it on the bridge's FEM before declaring anything.
- **C8 loadPattern** (`loadPattern.py`): `pattern Plain <step id> <tsTag> [-fact
  cFactor] { … }` containing, in order, the `load` conditions' lines, then `eleLoad`,
  `sp`, `genericLoad`, then `massToLoad` (C5). Per-node sums equal the rule to
  ≤ 9e-11 N in all 12 Tier-1 patterns. **V**
- **C9 UniformExcitation**: `pattern UniformExcitation <step id> <dir> -accel <tsTag>`,
  `direction` `dx..rz` → 1..6. **V**
- **C10 Time series**: `timeSeries Linear id` (`-factor` if set); `timeSeries Path id
  -dt dt -values {…} -factor cFactor [-startTime tStart if -startTime]`. The `.scd`
  holds a one-value placeholder (`list_of_values = (0.0,)`, `choice = constant`) in
  every San Ramon document; the campaign decks carry 24001-value records that are
  **not** in the `.scd`. Values come from `records={ts_id: values}` when given, else
  the placeholder with `placeholder=True`. **V** (arguments)
- **C11 Rayleigh**: `rayleigh alphaM/Rayleigh, Betak if Kcurr, Betak if Kinit, Betak if
  Kcomm` (1A: 0.38348… 0 0.0014944… 0; 1B–1D: 0.11968… 0 0.00060630… 0) →
  `ops.damping.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)`. **V**
- **C12 Stages**: each `Analyses.AnalysesCommand` closes a stage; the patterns defined
  by steps since the previous stage are added before it; `loadConst -time 0.0` after it
  when `loadConst` is true (1A: true ×4 then false; 1B–1D: true ×5). Static stages:
  `integrator LoadControl (duration/numIncr)`, `numIncr` increments (2 in Tier 1).
  Emit static stages with `ops.stage(name)`: `s.pattern(series=…)` + loads,
  `s.analysis(...)` mapped from STKO's choices, `s.run(n_increments=numIncr)`. The
  transient stage (Newmark γ β, adaptive driver, `duration/transient`, `numIncr`) is
  returned as a `StageSpec` and not emitted. **V** (structure)
  The test (`test.py writeTcl_test`, `AnalysesCommand.py`): `test <cmd> tol iter
  [pFlag if use_pFlag] [nType if use_nType]`, with `iter = 2 × iter/<cmd>` when
  `Adaptive Time Step` is on; `chain["test"]` stores `max_iter` as written,
  `desired_iter` (the adaptive driver's target) apart, `print_flag` (0 unless
  `use_pFlag`) and `n_type`. Tier-1 transient tests: 1A `NormUnbalance 1e-4 10`; 1B
  `NormDispIncr 1e-4 100` (iter 50); 1C, 1D `NormDispIncr 1e-4 400` (iter 200) —
  equal to the decks. A static stage after a transient one is refused (static
  stages are emitted before the time-history driver runs).
- **C13 IMPL-EX dTime targets** (`time_increment_utils.py`): elements on sub-shapes
  whose physical property is, or references (transitively, `property_references`), an
  `ASDConcrete1D/3D`, `DamageTC1D/3D`, `ASDBondSlip` or `ASDSteel1D` with `integration
  == "IMPL-EX"` or `eta != 0`. STKO lists the elements by *physical* assignment, so it
  also lists non-analysis edge elements (1D lists 18038 ids; 11688 are analysis
  elements): keep analysis elements only. Matches 984 (1B), 2232 (1C), 11688 (1D);
  1A has none. **V**
  **The reset** (`STKO_DT_UTIL_OnBeforeAnalyze`, written into every Analyses
  template): before every increment STKO calls `setParameter -val $dt -ele $e dTime`
  on every target, and at increment 1 of the stage also `dTimeCommit` and
  `dTimeInitial`; `$dt = duration/numIncr` (constant within a non-adaptive stage).
  `build_conditions(implex_dtime=True)` emits the same at the start of every static
  stage: `s.update_parameter(name, duration/numIncr, elements=targets)` for
  `dTimeCommit`, `dTimeInitial`, `dTime` (bridge: `parameter` / `addToParameter …
  element e <name>` / `updateParameter` / `remove parameter`, before the stage's
  analyze). Why it matters: without it, ASDConcrete reads OpenSees' own increment
  (`ops_Dt`, set in `Domain::applyLoad` to the pseudo-time step;
  `ASDConcrete3DMaterial.cpp`, `if (!dtime_is_user_defined) dtime_n = ops_Dt`) and
  carries the previous stage's committed increment, so the IMPL-EX factor
  `dtime_n / dtime_n_commit` at a stage's first increment is `dt_new/dt_old` instead
  of STKO's 1 — and 1B/1C mix 0.2 and 0.5 between static stages. Once set,
  `dtime_is_user_defined` stays true: **the time-history driver must set `dTime`
  before every transient step** (and `dTimeCommit`/`dTimeInitial` at its first), as
  STKO's transient template does, or the materials keep the last static increment.
  `implex_dtime=False` writes no reset. Smoke on 1B–1D: §11.
- **C14 Ignored steps**: recorders, regions, monitors, `customCommand` (payload = its
  `TCLscript`; San Ramon's are `EnergyBalance`, `getMass`, `Exit`), and
  `ImplexAutoErrorControlActivate` **only when no Analyses command follows it** (in
  1B–1D it is the last step, written after `exit`): the same position rule as a
  pattern after the last analysis. Before an analysis it sets up the IMPL-EX error
  control of that analysis (`ImplexAutoErrorControlActivate.py`) and is refused
  (option `…:before analysis`). Listed in `ConditionsPlan.ignored` and
  `TranslateResult.ignored`.

## 6. Work package M — `translate_mesh.py` (+ reader fixes)

**Owns:** `src/apeGmsh/interop/stko/translate_mesh.py`,
`src/apeGmsh/interop/stko/{model.py,read_scd.py}` (fixes only, extend, do not
rewrite), `tests/interop/test_stko_translate_mesh.py`. Creates `translate_types.py`
(§3) if it is missing.

```python
def check_supported(scd: ScdModel) -> list[Unsupported]: ...
def build_mesh(g, scd: ScdModel) -> MeshMap: ...
def shell_node_order(scd: ScdModel, eid: int) -> tuple[int, ...]: ...   # rule M3
```

Steps of `build_mesh` (the full-scale spike `spike/spike_full.py` does all of them):

1. **Diaphragm groups first.** For each `Constraints.mp.rigidDiaphragm` condition, for
   each of its interactions, for each link element: master `nodes[0]`, slaves
   `nodes[1:]`; group by master (link order kept). For each master create two
   discrete point entities: the master carrier (owns the master node) and the slave
   carrier (owns the slaves), each with one point element (gmsh type 15) per node,
   synthetic ids from `max(scd.mesh.elements) + 1` upward. PGs `rd:{cid}:{master}:master`
   and `rd:{cid}:{master}:slaves`. Ownership comes first because `g.constraints`
   resolves nodes by `getNodes` (classification); a carrier that does not own its
   nodes resolves to zero nodes (first spike attempt failed exactly so).
2. **Referenced sub-shapes**: analysis-element sub-shapes, every condition's
   `geometry` sub-shapes, every selection set's items (`whole` geometries expanded to
   all their sub-shapes). Process in order vertices, edges, faces, solids (this order
   is the node-ownership priority after the carriers).
3. **One discrete entity per sub-shape** (`gmsh.model.addDiscreteEntity(dim)`), split
   by `local_axis` when its analysis elements disagree (rule M6). Vertex: one point
   element on `mesh.vertex_nodes[geom][i]`. Edge/face/solid: all of
   `mesh.domains[(geom, kind)][i]` with their **STKO ids**, shells in `shell_node_order`.
   Nodes not yet owned are added (`gmsh.model.mesh.addNodes`) to the entity that first
   references them, with STKO's coordinates. Element type map: line2 → 1, quad4 → 3.
4. **PGs** with `g.physical.add(dim, tags, name=...)`:
   - element groups: key `(element property, physical property, local_axis)`; name
     `f"{ep.name}|{pp.name}"`, with `f"|L{j}"` (j from 1, by sorted `local_axis`) when one
     (ep, pp) has several axes, and `f"{ep.name}#{ep.id}|{pp.name}#{pp.id}"` if two
     pairs would get the same name. `ElementGroup` per group.
   - selection sets: the set name when the set has one kind, else `f"{name}.{kind}"`.
   - conditions: `f"cond:{cid}:{name}"`, plus `f".{kind}"` when it has several kinds.
5. Return the `MeshMap`. Do **not** call `generate`, `renumber`, `remove_orphans`, or
   anything that re-meshes; do not touch OCC.

Reader fixes owned here: `read_scd._read_local_axes` skips every `LOCAL_AXES/LOCAX_n`
because they are **datasets with attributes**, not groups (1B's `rotatedWalls`, 12
values `0 -3e4 0  0 -1 0  1 0 0  0 0 1`); read datasets. Document `SubShapeRef.kind`
codes 1/2/3 = vertex/edge/face in `model.py`.

`check_supported` (mesh slice): mesh element types of analysis elements; interactions
(NN with rigidDiaphragm = S, else T); the option rows of §4 that are mesh-level.

**Done when:** for 1A–1D the session's FEM nodes, every element group's elements
(ids, connectivity, node order) and every condition/selection-set PG node set equal
the rules (spike numbers: nodes 14003 / 2547 / 3979 / 13459; elements 13704 / 2456 /
3704 / 13160; 1A: 4381 entities, 68 PGs, 9175 carriers, under 5 s to `get_fem_data`).

## 7. Work package P — `translate_props.py`

**Owns:** `src/apeGmsh/interop/stko/translate_props.py`,
`tests/interop/test_stko_translate_props.py`.

```python
#: XOBJ_META -> handler; status "supported" | "tier_only".
ELEMENT_HANDLERS: dict[str, ElementHandler]
PROPERTY_HANDLERS: dict[str, PropertyHandler]

def check_supported(scd: ScdModel) -> list[Unsupported]: ...
def build_props(ops, scd: ScdModel, groups: Sequence[ElementGroup]) -> PropsResult: ...
```

The handler classes are yours to shape; keep them a dict keyed by `XOBJ_META` so a
later type is one entry. Requirements:

- Build each reachable physical property **once** (memoised by STKO id), dependencies
  first (`property_references`). Rules P5–P10. Register raw primitives with
  `ops.register(...)` when the namespace has no matching constructor.
- One element declaration per `ElementGroup`: rules P1–P4, `pg=group.pg`, transform
  shared per distinct `local_axis` (rule P4).
- `BeamSectionProperty` is not a primitive: it resolves to the integration of P3.
- sections.Elastic used by elasticBeamColumn is not emitted (P2).
- `check_supported` (props slice): element properties of analysis elements and the
  physical-property closure; §4 statuses; the props option rows of §4.

**Done when:** for each Tier-1 case, every reachable property's emitted values equal
§5 (P9 backbones and `lch_ref` to relative 1e-12; P8 fibers exact as a multiset) and
every element group emits with the right section/integration/transform; synthetic
tests cover each handler and each unsupported option.

## 8. Work package C — `translate_conditions.py`

**Owns:** `src/apeGmsh/interop/stko/translate_conditions.py`,
`tests/interop/test_stko_translate_conditions.py`.

```python
def check_supported(scd: ScdModel) -> list[Unsupported]: ...
def plan_conditions(scd: ScdModel, *, records: Mapping[int, Sequence[float]] | None = None) -> ConditionsPlan: ...
def diaphragm_groups(scd: ScdModel) -> tuple[DiaphragmGroup, ...]: ...   # C7, shared with the mesh module
def declare_session(g, scd: ScdModel, plan: ConditionsPlan, mesh: MeshMap) -> None: ...
def resolved_diaphragm_pairs(fem) -> list[tuple[int, int, int]]: ...
def verify_diaphragms(fem, plan: ConditionsPlan) -> None: ...
def build_conditions(ops, plan: ConditionsPlan, mesh: MeshMap, *, stages: bool = True,
                     chain: Literal["serial", "stko"] = "serial", implex_dtime: bool = True) -> dict[int, Any]: ...
```

- `plan_conditions` is pure (no gmsh, no bridge): rules C1–C13 over the `ScdModel`
  (lumping per element with the mesh coordinates; `quat_matrix` for C2/C3 local
  forces). Fills one `ConditionSummary` per condition with `pgs={}`: the plan is
  built without the session, so the integrator fills `pgs` from
  `mesh.condition_pgs` (`dataclasses.replace`) when it assembles `TranslateResult`
  (§9). Nothing in `build_conditions` reads `pgs`.
- `declare_session`: one `g.constraints.rigid_diaphragm(d.master_pg, d.slave_pg,
  master_point=xyz(master), plane_normal=e_{perp}, constrained_dofs=…,
  plane_tolerance=2·max slave offset + 1e-6·max(1, bbox diagonal),
  name="rd:{cid}:{master}")` per `DiaphragmGroup` (C7: the tolerance keeps every
  slave; no caller-chosen tolerance). 360 / 840 pairs identical to STKO; the
  resolved records are asserted equal to the plan by `verify_diaphragms`.
- `build_conditions`:
  - `ops.fix(nodes=..., dofs=mask)` per mask (C6);
  - `ops.mass(nodes=[n], values=m)` per node with mass (C1);
  - time series: `ops.timeSeries.Linear(...)`, `ops.timeSeries.Path(values=, dt=,
    factor=, start_time=)` (C10);
  - `ops.damping.rayleigh(...)` (C11);
  - first, on a real bridge, `verify_diaphragms(ops.fem, plan)`;
  - with `stages=True`: one `ops.stage(name)` per static `StageSpec`, holding the
    IMPL-EX reset of C13 (`implex_dtime=True`), its
    patterns (`s.pattern(series=ts)`, `p.load(node=n, forces=v)` per node), its
    analysis chain (constraints `Auto(auto_penalty_oom=oom)`, numberer, system, test,
    algorithm, `LoadControl(dlam=duration/n_incr)`, `Static()`), and
    `s.run(n_increments=n_incr)`; `stages=False` emits all Plain patterns globally
    (deck parity of loads only). UniformExcitation patterns are created
    (`ops.pattern.UniformExcitation(direction, series)`) only when the caller asks for
    the transient stage — not in v1.
- `check_supported` (conditions slice): §4 rows owned by conditions.

**Done when:** for each Tier-1 case the plan equals §5 numbers (masses, loads per
pattern, fix sets, diaphragm pairs, time-series arguments, Rayleigh, stage structure,
dTime targets) and the emitted deck passes the §9.3 categories for these items.

## 9. Integrator — `translate.py`, the bridge option, `deck_parity.py`

**Owns:** `src/apeGmsh/interop/stko/translate.py`, `src/apeGmsh/interop/stko/deck_parity.py`,
the exports in `src/apeGmsh/interop/stko/__init__.py`, the bridge option (§9.1) with
its tests under `tests/opensees/`, `tests/interop/test_stko_translate.py`,
`tests/interop/test_stko_deck_parity.py`, `tests/interop/test_stko_oracles.py`, and the
CHANGELOG fragment `changelog.d/2026-10-02-stko-translator.md`.

```python
def collect_unsupported(scd: ScdModel) -> tuple[Unsupported, ...]: ...
def translate_scd(g, scd: ScdModel | str | Path, *, records: Mapping[int, Sequence[float]] | None = None) -> TranslateResult: ...
def build_opensees(fem, result: TranslateResult, *, element_tags: Literal["fem", "sequential"] = "fem", stages: bool = True): ...
```

`collect_unsupported` concatenates the three `check_supported` lists, merges entries
with the same `(category, xobj_meta, reason)` (ids/names unioned) and sorts them.
`translate_scd` fills each `ConditionSummary.pgs` from `mesh.condition_pgs`.

### 9.1 Bridge option `element_tags`

`apeSees(fem, *, element_tags: Literal["sequential", "fem"] = "sequential")`. With
`"fem"`, `allocate_element_tags` builds each spec's `ElementPlanRows(eids, conn, 0,
tags=eids)` and advances the `"element"` counter to `max(eids)`; node-pair specs
(`MISSING_FEM_ELEMENT_ID`) keep `allocate_block`. Audit every other
`allocate("element")` site (staged pre-allocation, partitioned pre-allocation,
synthesised springs/links) so nothing is allocated below the FEM range, and refuse
`"fem"` when a FEM eid is ≤ 0 or duplicated. The scratch monkeypatch (≈15 lines,
`spike/spike_fem_tags.py`) gave 13704/13704 tags equal to STKO's. Follow the
bridge-feature skill (lock tests, `static-gates`). **Implemented 2026-10-02**
(`build.reserve_fem_element_tags`, `allocate_element_tags(..., element_tags=)`,
`TagAllocator.reserve_through`; the counter moves past the snapshot's largest
element id, not only the emitted ones). `build_opensees` uses it by default; the
Tier-1 decks carry STKO's element ids (1A 13704, 1B 2456, 1C 3704, 1D 13160).

### 9.2 What the integrator verifies end to end

For each Tier-1 case: `translate_scd` + `build_opensees` + `ops.tcl`, then
`deck_parity.compare(stko_deck_dir, our_deck)` → every gated row of §9.3 passes; for
1A also an eigen run (6 modes) within 0.1 % of STKO's T1–T6 (spike: 1.66477 vs 1.6648
s). 1B/1C/1D: the campaign decks carry real records, so pass `records=` parsed from
their `definitions.tcl` to compare time series values.

### 9.3 Parity contract

`canonical_deck(path) -> CanonicalDeck` reads either dialect: STKO's partitioned
export (a directory with `main.tcl` and its sourced files) or one apeSees Tcl file.

**Canonicalisation.**
- Join backslash continuations; drop comments; ignore `if {$STKO_VAR_process_id == k}`
  structure (every partition block is read).
- Nodes: union by tag; repeated copies must have identical coordinates (else a
  `conflict`). Mass: from `node … -mass` (STKO) or `mass n …` (apeSees); at most one
  copy may carry it.
- Elements: union by tag; repeated copies identical. Resolve the transform tag to its
  `(type, vecxz, options)`, the section tag to its signature, inline STKO integration
  `Type sec np` and apeSees `beamIntegration Type tag sec np` to the same
  `(Type, section signature, np)`.
- Signatures: a section/material is `(command, type, numeric args, referenced
  signatures)` with tags replaced by the referenced object's signature, recursively;
  a fiber section's fibers form a sorted multiset of `(y, z, A, material signature)`.
- Defaults are filled before comparing: ASDConcrete `-rho 0 -eta 0 -Kc 2/3 -cdf 0
  -implexAlpha 1.0`; Hysteretic `beta 0`; ElasticMembranePlateSection `Ep_mod 1`.
- `fix`: union per node, repeated masks identical. `rigidDiaphragm`: set of `(perp,
  master, slave)`.
- Loads: per pattern, the sum of every `load` line per node (all partitions).
  Patterns are matched by order among Plain patterns, then UniformExcitation by
  order; their time series by signature `(type, dt, factor, startTime, values)`.
- Stages: the command sequence between `domainChange` (STKO) or stage brackets
  (apeSees): patterns first active, `analysis` type, `integrator` with arguments,
  increments (`initial_num_incr` / `analyze n`), `loadConst`.
- Excluded from comparison: recorders, regions, monitors, custom procs, `exit`,
  `wipe*`, the dTime/IMPL-EX procs, MPI `barrier`/`send`/`recv`, `getPID` plumbing.

**Comparison (gated unless marked info).** `x` is the STKO value.

| Category | Rule | Tolerance |
|---|---|---|
| node ids | set equality | exact |
| node coordinates | per node | `|Δ| ≤ 1e-9·max(1, bbox diagonal)` (STKO rounds to ~10 digits) |
| element ids | our tags == STKO's ids (set equality; the deck uses `element_tags="fem"`) | exact |
| connectivity bijection | the element with a node set carries the same tag in both decks; no node set on two elements | exact |
| element type, connectivity incl. node order | per element, by tag | exact |
| shell flags (`-corotational -noeas -drillingStab v -drillingNL`) | per element | exact (`v` as printed) |
| shell `-local` | per element | `|Δ| ≤ 1e-6` (STKO `%.6g`) |
| beam transform type, vecxz, options | per element | type/options exact, `|Δ| ≤ 1e-11` |
| element numeric args (elasticBeamColumn A E G J Iy Iz; integration np) | per element | relative 1e-12; integers exact |
| section / material signature, reachable closure only | per element's section graph | numbers relative 1e-12, integers exact, fibers as multiset with `|Δy|,|Δz| ≤ 1e-9·section size`, `|ΔA|/A ≤ 1e-12` |
| nodal mass | per node, 6 components | `max|Δ| ≤ 1e-9·max|m|` of that node (floor `1e-15·max nodal mass`); totals relative 1e-9 |
| loads | per (pattern, node), 6 components | `max|Δ| ≤ 1e-9·max|F|` of that node (floor `1e-15·max nodal load of the pattern`); resultant relative 1e-9 |
| fix | node → mask | exact |
| rigidDiaphragm | pair set, deck and `plan.diaphragm_pairs` | exact |
| time series | type, dt, factor, startTime | exact |
| time-series values | per value | **info**: a pass-through (`records=` is parsed from STKO's own `definitions.tcl`) |
| Rayleigh | 4 coefficients | relative 1e-12 |
| UniformExcitation | direction, series signature | exact |
| stages | count, patterns per stage, analysis type, integrator + args, increments, loadConst | exact |
| solver chain (constraints, numberer, system, test, algorithm) | per static stage | **info** (STKO's MPI choices are not ours to copy) |
| transient solver chain (plan) | constraints, numberer, system, test (type, tol, `max_iter` as STKO writes it, flags), algorithm + options rebuilt from `StageSpec.chain` | exact |
| IMPL-EX dTime targets ∩ analysis elements | set, plan vs STKO | exact |
| IMPL-EX dTime reset (translated deck) | per static stage: `dTimeCommit`, `dTimeInitial`, `dTime` on the plan's targets, value = the stage's increment | exact |
| unreachable template properties, extra transforms/integration objects | — | not compared |

`compare(...) -> ParityReport` returns one row per category with pass/fail, counts,
max error and the first offenders; `ParityReport.ok` is true when every gated row
passes. The prototype `design/proto/verify.py` is the rules-side twin of this
comparison and its numbers are the targets.

## 10. Tests and data

- **CI tests are synthetic.** Build small `.scd` documents in `tmp_path`, extending the
  writer in `tests/interop/test_stko_reader.py` (`_write_scd`) inside your own test
  file (do not edit the reader's test). Cover every handler, every rule with a
  hand-computable case (a 2×1 quad mesh for consistent lumping, a rotated-shell case
  for M3, a link pair for C7), every unsupported type and option (the error lists all
  of them at once), and the "never silently skipped" contract.
- **Oracle tests run locally.** The Tier-1 tests of the translator modules (mesh,
  props, integrator) run when `APEGMSH_STKO_ORACLES` points at `models/stko-rev0-tcl`
  and skip otherwise (say so in the skip reason; CI never has the data). One
  variable for all of them, the reader's own San Ramon tests (`test_stko_reader.py`)
  included (they used `APEGMSH_STKO_SAMPLES` before 2026-10-02). Deck readers open
  STKO's files by name (`main.tcl`,
  `nodes.tcl`, `elements.tcl`, `materials.tcl`, `sections.tcl`, `definitions.tcl`,
  `analysis_steps.tcl`), never a glob: a translated deck may sit next to them.
- `pytest tests/interop -q -p no:cacheprovider` from the worktree;
  `python scripts/check_quirks.py`; the bridge option also runs `static-gates` and the
  lock tests its skill names.

## 11. Not done, open, deviations

- **Transient stage and STKO's adaptive driver** are data (`StageSpec`), not emitted;
  the San Ramon TH scripts own them. The static stages reset the IMPL-EX `dTime`
  of the C13 targets as STKO does; after that the materials no longer follow
  OpenSees' increment, so **the TH driver must apply the per-step `setParameter
  dTime`** (and `dTimeCommit`/`dTimeInitial` at its first step). With
  `implex_dtime=False` no reset is written and the materials follow `ops_Dt`.
- **Time-series values** are placeholders unless `records=` is passed.
- **Hypotheses not exercised by Tier 1:** C2/C3 local (`Global = False`) forces;
  AutoEdgeMass area with two sections in the closure; geomTransf PDelta/Corotational;
  `-iter`, `-mass` on beams.
- **Unverified:** `g.save()` / `model.h5` of a translated session (discrete entities,
  carrier elements); partitioned emit (`ops.tcl(..., partitions=)`) of a translated
  model; performance beyond 1A size (4D has 91,655 mesh entries).
- **Tier 2–4 types** fail loudly (§4) until each gets a registry entry and a rule.
- **Iyy / Izz of a non-square `sections.Elastic`** (P2) is refused until a case or
  STKO's `MpcBeamSection` serialisation decides the `PROPS` order.
- **Known numeric deviations** (all within the §9.3 tolerances; none gated as
  exact):
  - beam `vecxz` is grouped after rounding to 12 significant digits (`local_axis`);
    the emitted value is that rounded one (Tier 1: |Δ| ≤ 2.2e-16 against STKO);
  - shell `-local` is STKO's `%.6g`, reproduced (|Δ| = 0);
  - a damage value STKO computes as `-2.2e-16` (round-off of `1 - s/q`, 1D's
    `ASDConcrete1D` 9 and 10) is written `0.0` (the bridge refuses negative damage);
    the parity check accepts it through an absolute floor of 3e-16 on one fiber
    section (1D's 11);
  - node coordinates are the `.scd`'s full doubles; STKO prints them rounded
    (max |Δ| 4.3e-6 mm on a 6.8e4 mm model);
  - our test line carries `pFlag 0 nType 2` explicitly where STKO writes neither
    (both OpenSees defaults; solver chain is info).
