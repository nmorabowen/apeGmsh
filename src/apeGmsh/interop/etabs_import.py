"""Import a neutral StructuralModel into an apeGmsh session (Phase 2).

Builds geometry (points + lines) and solver-facing physical groups from an
analytical ETABS model, then — after the caller meshes — wires an ``apeSees``
deck of beam-column elements, fixities, and nodal loads.

Pipeline (see apeETABS ADR 0009 / build-plan W3 Phase 2):

    model = StructuralModel.from_json("two_story_frame.sm.json")
    with apeGmsh(model_name="frame") as g:
        result = import_structural_model(g, model)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=1)
        g.mesh.partitioning.renumber(dim=1, method="rcm", base=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = build_opensees(fem, model, result)
    ops.tcl("frame.tcl").py if hasattr(...) else ops.py("frame.py")

Phase 2 covers frames only. Areas, diaphragms, and distributed (frame/area)
loads are recorded in ``result.skipped`` for the later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apeGmsh.opensees import apeSees

from .model import StructuralModel

# A unit vecxz for the geomTransf local x-z plane, keyed by member orientation.
# Vertical members (axis ~ Z) take (1,0,0); everything else takes (0,0,1).
# Both are guaranteed non-parallel to their member axis for the cases they tag.
_VECXZ = {"v": (1.0, 0.0, 0.0), "h": (0.0, 0.0, 1.0)}


@dataclass(frozen=True, slots=True)
class FrameGroup:
    """One emit unit: all line members of a (section, orientation) sharing a PG."""
    pg: str
    section: str
    orient: str  # "v" | "h"


@dataclass(frozen=True, slots=True)
class AreaGroup:
    """One emit unit: all shell surfaces of a section sharing a PG."""
    pg: str
    section: str


@dataclass(frozen=True, slots=True)
class RestraintGroup:
    pg: str
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NodalLoadGroup:
    pg: str
    pattern: str
    forces: tuple[float, float, float, float, float, float]


@dataclass
class ImportResult:
    """Metadata bridging the pre-mesh build to the post-mesh OpenSees emit."""
    node_tag: dict[str, int]
    frame_groups: list[FrameGroup] = field(default_factory=list)
    area_groups: list[AreaGroup] = field(default_factory=list)
    restraint_groups: list[RestraintGroup] = field(default_factory=list)
    nodal_load_groups: list[NodalLoadGroup] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _orient(model: StructuralModel, fr) -> str:
    ni, nj = model.node(fr.i), model.node(fr.j)
    dz = abs(nj.z - ni.z)
    dx, dy = nj.x - ni.x, nj.y - ni.y
    horiz = (dx * dx + dy * dy) ** 0.5
    return "v" if dz > horiz else "h"


def import_structural_model(g, model: StructuralModel) -> ImportResult:
    """Build points, lines, and physical groups; return emit metadata."""
    gm = g.model.geometry

    # 1. Nodes -> points (etabs id -> gmsh point tag).
    node_tag: dict[str, int] = {
        n.id: gm.add_point(n.x, n.y, n.z) for n in model.nodes
    }

    # Shared-edge map: every undirected node-pair gets exactly ONE line, reused
    # by frames AND area boundaries. Surfaces built from these shared line tags
    # are conformal with the frames and with each other by construction (OCC
    # topology) — no fragment / weld step needed.
    edges: dict[frozenset, tuple[int, tuple[str, str]]] = {}

    def edge(a: str, b: str) -> tuple[int, tuple[str, str]]:
        key = frozenset((a, b))
        rec = edges.get(key)
        if rec is None:
            rec = (gm.add_line(node_tag[a], node_tag[b]), (a, b))
            edges[key] = rec
        return rec

    def curve_loop(loop_nodes: list[str]) -> int:
        signed: list[int] = []
        n = len(loop_nodes)
        for k in range(n):
            a, b = loop_nodes[k], loop_nodes[(k + 1) % n]
            tag, stored = edge(a, b)
            signed.append(tag if stored == (a, b) else -tag)
        return gm.add_curve_loop(signed)

    # 2. Frames -> shared lines, bucketed by (section, orientation).
    buckets: dict[tuple[str, str], list[int]] = {}
    for fr in model.frames:
        tag, _ = edge(fr.i, fr.j)
        buckets.setdefault((fr.section, _orient(model, fr)), []).append(tag)

    # A section is split into "<sec>__v"/"<sec>__h" only when it spans both
    # orientations; otherwise the PG is just the section name.
    sections_with_both = {
        sec for sec in {k[0] for k in buckets}
        if (sec, "v") in buckets and (sec, "h") in buckets
    }
    frame_groups: list[FrameGroup] = []
    for (sec, orient), tags in buckets.items():
        pg = f"{sec}__{orient}" if sec in sections_with_both else sec
        g.physical.add_curve(tags, name=pg)
        frame_groups.append(FrameGroup(pg=pg, section=sec, orient=orient))

    # 2b. Areas -> planar surfaces from shared edges, bucketed by section.
    area_buckets: dict[str, list[int]] = {}
    for ar in model.areas:
        surf = gm.add_plane_surface([curve_loop(list(ar.nodes))])
        area_buckets.setdefault(ar.section, []).append(surf)
    area_groups: list[AreaGroup] = []
    for sec, tags in area_buckets.items():
        g.physical.add_surface(tags, name=sec)
        area_groups.append(AreaGroup(pg=sec, section=sec))

    # 3. Restraints -> point PGs, one per distinct DOF mask.
    by_mask: dict[tuple[int, ...], list[int]] = {}
    for r in model.restraints:
        by_mask.setdefault(r.dofs, []).append(node_tag[r.node])
    restraint_groups: list[RestraintGroup] = []
    for mask, tags in by_mask.items():
        pg = "fix_" + "".join(str(b) for b in mask)
        g.physical.add_point(tags, name=pg)
        restraint_groups.append(RestraintGroup(pg=pg, dofs=mask))

    # 4. Nodal loads -> one point PG per (pattern, node).
    nodal_load_groups: list[NodalLoadGroup] = []
    for pat in model.loads:
        for ld in pat.nodal:
            pg = f"load_{pat.name}_{ld.node}"
            g.physical.add_point([node_tag[ld.node]], name=pg)
            nodal_load_groups.append(
                NodalLoadGroup(pg=pg, pattern=pat.name, forces=ld.forces6)
            )

    # 5. Record what this phase does not yet consume.
    skipped: list[str] = []
    if model.diaphragms:
        skipped.append(f"{len(model.diaphragms)} diaphragm(s) — deferred to Phase 4")
    n_dist = sum(len(p.frame) + len(p.area) for p in model.loads)
    if n_dist:
        skipped.append(f"{n_dist} distributed load(s) — eleLoad deferred to Phase 4")

    return ImportResult(
        node_tag=node_tag,
        frame_groups=frame_groups,
        area_groups=area_groups,
        restraint_groups=restraint_groups,
        nodal_load_groups=nodal_load_groups,
        skipped=skipped,
    )


def build_opensees(fem, model: StructuralModel, result: ImportResult,
                   *, ndm: int = 3, ndf: int = 6,
                   shell_element: str = "ASDShellT3"):
    """Wire an apeSees deck from a meshed FEM snapshot + import metadata.

    Returns the ``apeSees`` object (not yet exported) so the caller chooses
    ``ops.tcl(path)`` / ``ops.py(path)``.

    ``shell_element`` selects the 3-node shell type for area surfaces (the mesh
    is triangular); pass a 4-node type only if you recombined to quads.
    """
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndf)

    # One geomTransf per orientation actually used.
    transf = {
        orient: ops.geomTransf.Linear(vecxz=_VECXZ[orient])
        for orient in {fg.orient for fg in result.frame_groups}
    }

    for fg in result.frame_groups:
        sec = model.section(fg.section)
        mat = model.material(sec.material) if sec.material else None
        if mat is None:
            raise ValueError(f"section {fg.section!r} has no resolvable material")
        p = sec.props
        kw = dict(A=p["A"], E=mat.E, Iz=p["Iz"])
        if ndm == 3:
            kw.update(Iy=p["Iy"], G=mat.G, J=p["J"])
        ops.element.elasticBeamColumn(pg=fg.pg, transf=transf[fg.orient], **kw)

    # Shells -> elastic membrane-plate section (no separate nDMaterial needed).
    shell_ctor = getattr(ops.element, shell_element)
    for ag in result.area_groups:
        sec = model.section(ag.section)
        mat = model.material(sec.material) if sec.material else None
        if mat is None:
            raise ValueError(f"section {ag.section!r} has no resolvable material")
        if sec.thickness is None:
            raise ValueError(f"shell section {ag.section!r} has no thickness")
        plate = ops.section.ElasticMembranePlateSection(
            E=mat.E, nu=mat.nu, h=sec.thickness, rho=mat.rho or 0.0,
        )
        shell_ctor(pg=ag.pg, section=plate)

    for rg in result.restraint_groups:
        ops.fix(pg=rg.pg, dofs=tuple(rg.dofs[:ndf]))

    if result.nodal_load_groups:
        ts = ops.timeSeries.Linear()
        by_pattern: dict[str, list[NodalLoadGroup]] = {}
        for lg in result.nodal_load_groups:
            by_pattern.setdefault(lg.pattern, []).append(lg)
        for name, groups in by_pattern.items():
            with ops.pattern.Plain(series=ts) as pat:
                for lg in groups:
                    pat.load(pg=lg.pg, forces=tuple(lg.forces[:ndf]))

    return ops
