"""Solve a neutral StructuralModel and return joint-keyed results.

A convenience wrapper over the full interop pipeline
(``import_structural_model`` -> mesh -> ``apply_subgrade_springs`` ->
``build_opensees``) that runs a single static load case in openseespy and
reports nodal **displacements** and support **reactions** keyed by the original
ETABS joint id. This is the apeGmsh half of the ADR 0009 solve cross-check:
its output aligns 1:1 (by joint id) with apeETABS' own analysis results.

Only the meshed nodes that coincide with an input joint are reported — interior
mesh nodes have no ETABS id and are skipped. Requires ``openseespy``.
"""
from __future__ import annotations

import runpy
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from .etabs_import import (
    apply_subgrade_springs,
    build_opensees,
    import_structural_model,
)
from .model import StructuralModel

Vec6 = tuple[float, float, float, float, float, float]


@dataclass
class SolveResult:
    """Joint-keyed static results for one load case (report units = model units).

    ``displacements`` covers every input joint that survived to the mesh;
    ``reactions`` covers the supported joints (restraints + spring grounds).
    Both map ``etabs_joint_id -> (1,2,3,4,5,6)`` = (Ux,Uy,Uz,Rx,Ry,Rz) /
    (Fx,Fy,Fz,Mx,My,Mz).
    """
    case: str
    displacements: dict[str, Vec6]
    reactions: dict[str, Vec6]
    converged: bool
    n_mesh_nodes: int


def solve_and_extract(
    model: StructuralModel | str | Path,
    *,
    case: str | None = None,
    global_size: float = 1.0,
    ndm: int = 3,
    ndf: int = 6,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> SolveResult:
    """Solve one static load ``case`` and return joint-keyed disp + reactions.

    Args:
        model: a :class:`StructuralModel` or a path to a ``.sm.json``.
        case: the load pattern to apply alone; defaults to the first pattern.
            Only this case is emitted, so the result is that case in isolation
            (no superposition) — matching a single ETABS output case.
        global_size: mesh size; finer = closer to the analytical field.
        tol, max_iter: ``NormDispIncr`` test tolerance / iteration cap.
    """
    import numpy as np  # noqa: F401 (kept local; heavy optional dep chain)

    if isinstance(model, (str, Path)):
        model = StructuralModel.from_json(model)

    pattern = _resolve_case(model, case)
    # Emit only the chosen case so the solve is that case in isolation.
    one_case = replace(model, loads=[p for p in model.loads if p.name == pattern])

    dim = 2 if one_case.areas else 1
    from apeGmsh import apeGmsh

    g = apeGmsh(model_name="xcheck", verbose=False)
    g.begin()
    try:
        result = import_structural_model(g, one_case)
        if dim == 2:
            g.mesh.sizing.set_global_size(global_size)
            g.mesh.generation.generate(dim=2)
        else:
            for fg in result.frame_groups:
                g.mesh.structured.set_transfinite_curve(fg.pg, n_nodes=2)
            g.mesh.generation.generate(dim=1)
        g.mesh.partitioning.renumber(dim=dim, base=1)
        apply_subgrade_springs(g, one_case, result)
        fem = g.mesh.queries.get_fem_data(dim=None)
    finally:
        g.end()

    ops = build_opensees(fem, one_case, result, ndm=ndm, ndf=ndf)
    converged = _run_static(ops, tol=tol, max_iter=max_iter)

    import openseespy.opensees as o

    tag_of = _joint_tag_map(fem, one_case)
    dofs = range(1, ndf + 1)
    displacements = {
        jid: tuple(float(o.nodeDisp(t, d)) for d in dofs)
        for jid, t in tag_of.items()
    }
    o.reactions()
    supported = {r.node for r in one_case.restraints} | {s.node for s in one_case.springs}
    supported |= {a for a in _area_spring_nodes(one_case)}
    reactions = {
        jid: tuple(float(o.nodeReaction(tag_of[jid], d)) for d in dofs)
        for jid in supported
        if jid in tag_of
    }
    return SolveResult(
        case=pattern,
        displacements=displacements,
        reactions=reactions,
        converged=converged,
        n_mesh_nodes=int(fem.info.n_nodes),
    )


def _resolve_case(model: StructuralModel, case: str | None) -> str:
    names = [p.name for p in model.loads]
    if not names:
        raise ValueError("model has no load patterns to solve.")
    if case is None:
        return names[0]
    if case not in names:
        raise ValueError(f"case {case!r} not in load patterns {names}.")
    return case


def _run_static(ops, *, tol: float, max_iter: int) -> bool:
    """Write + run the deck, configure a linear static step, return converged."""
    with tempfile.TemporaryDirectory() as d:
        deck = Path(d) / "deck.py"
        ops.py(str(deck))
        runpy.run_path(str(deck))
    import openseespy.opensees as o

    o.system("UmfPack")
    o.numberer("RCM")
    o.constraints("Transformation")
    o.integrator("LoadControl", 1.0)
    o.test("NormDispIncr", tol, max_iter)
    o.algorithm("Linear")
    o.analysis("Static")
    return o.analyze(1) == 0


def _joint_tag_map(fem, model: StructuralModel) -> dict[str, int]:
    """``etabs_joint_id -> fem node tag`` by coincident coordinates."""
    import numpy as np

    ids = np.asarray(fem.nodes.ids)
    coords = np.asarray(fem.nodes.coords, dtype=float)
    decoupled = set(int(i) for i in fem.nodes.decoupled_ids)
    key_to_tag = {
        tuple(np.round(c, 6)): int(i)
        for i, c in zip(ids, coords)
        if int(i) not in decoupled
    }
    out: dict[str, int] = {}
    for n in model.nodes:
        tag = key_to_tag.get(tuple(np.round(n.xyz, 6)))
        if tag is not None:
            out[n.id] = tag
    return out


def _area_spring_nodes(model: StructuralModel) -> set[str]:
    """Input joints that bound a subgrade-sprung area (they carry reactions)."""
    sprung = {a.area for a in model.area_springs}
    return {nid for ar in model.areas if ar.id in sprung for nid in ar.nodes}
