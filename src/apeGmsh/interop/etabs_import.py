"""Import a neutral StructuralModel into an apeGmsh session.

Builds geometry + physical groups from an analytical ETABS model, declares
loads and self-mass on the session, then — after the caller meshes — wires an
``apeSees`` deck (beam-column + shell elements, fixities, loads, mass, rigid
diaphragms).

Pipeline (see apeETABS ADR 0009):

    model = StructuralModel.from_json("wall_slab_frame.sm.json")
    with apeGmsh(model_name="m") as g:
        result = import_structural_model(g, model)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=2)          # 1 for frames-only models
        g.mesh.partitioning.renumber(base=1)
        fem = g.mesh.queries.get_fem_data(dim=None)
    ops = build_opensees(fem, model, result)
    ops.tcl("m.tcl"); ops.py("m.py")

Coverage:
- Phase 2  frames -> beam-column elements, fixities, nodal loads.
- Phase 3  areas  -> conformal shell surfaces (shared-edge topology).
- Phase 4  distributed frame/area loads (tributary nodal), self-mass from
           material density, rigid diaphragms (auto-skipped when a shell slab
           already provides the diaphragm action).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apeGmsh.opensees import apeSees

from .model import StructuralModel

# A unit vecxz for the geomTransf local x-z plane, keyed by member orientation.
# Vertical members (axis ~ Z) take (1,0,0); everything else takes (0,0,1).
_VECXZ = {"v": (1.0, 0.0, 0.0), "h": (0.0, 0.0, 1.0)}

# ETABS load direction token -> global unit axis.
_AXIS = {
    1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0),
    "X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0),
    "gravity": (0.0, 0.0, -1.0),
}


def _unit(direction) -> tuple[float, float, float]:
    return _AXIS[direction.strip() if isinstance(direction, str) else direction]


@dataclass(frozen=True, slots=True)
class FrameGroup:
    """All line members of a (section, orientation) sharing a PG."""
    pg: str
    section: str
    orient: str  # "v" | "h"


@dataclass(frozen=True, slots=True)
class AreaGroup:
    """All shell surfaces of a section sharing a PG."""
    pg: str
    section: str


@dataclass(frozen=True, slots=True)
class RestraintGroup:
    pg: str
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DiaphragmSpec:
    """A diaphragm joint set; ``shell_backed`` ones are skipped at build time."""
    name: str
    pg: str
    plane_normal: tuple[float, float, float]
    dofs: tuple[int, ...]
    shell_backed: bool


@dataclass
class ImportResult:
    """Metadata bridging the pre-mesh build to the post-mesh OpenSees emit."""
    node_tag: dict[str, int]
    frame_groups: list[FrameGroup] = field(default_factory=list)
    area_groups: list[AreaGroup] = field(default_factory=list)
    restraint_groups: list[RestraintGroup] = field(default_factory=list)
    load_patterns: list[str] = field(default_factory=list)
    diaphragms: list[DiaphragmSpec] = field(default_factory=list)
    has_masses: bool = False
    skipped: list[str] = field(default_factory=list)


def _orient(model: StructuralModel, fr) -> str:
    ni, nj = model.node(fr.i), model.node(fr.j)
    dz = abs(nj.z - ni.z)
    horiz = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
    return "v" if dz > horiz else "h"


def import_structural_model(g, model: StructuralModel, *,
                            self_mass: bool = True) -> ImportResult:
    """Build geometry + physical groups and declare loads / mass on ``g``.

    ``self_mass`` lumps mass from material density (line: rho*A, shell:
    rho*thickness). Has no effect on a static analysis; needed for modal /
    dynamic.
    """
    gm = g.model.geometry

    # 1. Nodes -> points.
    node_tag: dict[str, int] = {
        n.id: gm.add_point(n.x, n.y, n.z) for n in model.nodes
    }

    # Shared-edge map: every undirected node-pair -> ONE line, reused by frames
    # AND area boundaries. Surfaces built from these shared lines are conformal
    # with the frames and each other by construction (no fragment/weld step).
    edges: dict[frozenset, tuple[int, tuple[str, str]]] = {}

    def edge(a: str, b: str) -> tuple[int, tuple[str, str]]:
        key = frozenset((a, b))
        rec = edges.get(key)
        if rec is None:
            rec = (gm.add_line(node_tag[a], node_tag[b]), (a, b))
            edges[key] = rec
        return rec

    def curve_loop(loop_nodes: list[str]) -> int:
        signed = []
        n = len(loop_nodes)
        for k in range(n):
            a, b = loop_nodes[k], loop_nodes[(k + 1) % n]
            tag, stored = edge(a, b)
            signed.append(tag if stored == (a, b) else -tag)
        return gm.add_curve_loop(signed)

    # 2. Frames -> shared lines, bucketed by (section, orientation).
    frame_line: dict[str, int] = {}
    buckets: dict[tuple[str, str], list[int]] = {}
    for fr in model.frames:
        tag, _ = edge(fr.i, fr.j)
        frame_line[fr.id] = tag
        buckets.setdefault((fr.section, _orient(model, fr)), []).append(tag)
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
    area_surf: dict[str, int] = {}
    area_buckets: dict[str, list[int]] = {}
    for ar in model.areas:
        surf = gm.add_plane_surface([curve_loop(list(ar.nodes))])
        area_surf[ar.id] = surf
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

    # 4. Loads -> declared on g.loads, grouped by pattern (= load case). All
    #    forms resolve to tributary NODAL loads, emitted later via from_model.
    loaded_nodes = {ld.node for p in model.loads for ld in p.nodal}
    for nid in loaded_nodes:
        g.physical.add_point([node_tag[nid]], name=f"pt_{nid}")
    loaded_frames = {fl.frame for p in model.loads for fl in p.frame}
    for fid in loaded_frames:
        g.physical.add_curve([frame_line[fid]], name=f"frmload_{fid}")
    loaded_areas = {al.area for p in model.loads for al in p.area}
    for aid in loaded_areas:
        g.physical.add_surface([area_surf[aid]], name=f"srfload_{aid}")

    load_patterns: list[str] = []
    for pat in model.loads:
        if not (pat.nodal or pat.frame or pat.area):
            continue
        load_patterns.append(pat.name)
        with g.loads.case(pat.name):
            for ld in pat.nodal:
                if any(ld.force_xyz):
                    g.loads.point.force(pg=f"pt_{ld.node}", force=ld.force_xyz)
                if any(ld.moment_xyz):
                    g.loads.point.moment(pg=f"pt_{ld.node}", moment=ld.moment_xyz)
            for fl in pat.frame:
                g.loads.line(pg=f"frmload_{fl.frame}", magnitude=fl.value,
                             direction=_unit(fl.direction), target_form="nodal")
            for al in pat.area:
                u = _unit(al.direction)
                g.loads.surface.traction(
                    pg=f"srfload_{al.area}",
                    vector=tuple(c * al.value for c in u),
                    target_form="nodal")

    # 5. Self-mass from material density (line: rho*A, shell: rho*thickness).
    has_masses = False
    if self_mass:
        for fg in frame_groups:
            sec = model.section(fg.section)
            mat = model.material(sec.material) if sec.material else None
            if mat and mat.rho:
                g.masses.line(pg=fg.pg, linear_density=mat.rho * sec.props["A"])
                has_masses = True
        for ag in area_groups:
            sec = model.section(ag.section)
            mat = model.material(sec.material) if sec.material else None
            if mat and mat.rho and sec.thickness:
                g.masses.surface(pg=ag.pg, areal_density=mat.rho * sec.thickness)
                has_masses = True

    # 6. Diaphragms -> point PGs; shell-backed ones are flagged for skip.
    area_node_ids = {nid for ar in model.areas for nid in ar.nodes}
    diaphragms: list[DiaphragmSpec] = []
    for dp in model.diaphragms:
        pg = f"diaph_{dp.name}"
        g.physical.add_point([node_tag[n] for n in dp.nodes], name=pg)
        diaphragms.append(DiaphragmSpec(
            name=dp.name, pg=pg, plane_normal=(0.0, 0.0, 1.0), dofs=(1, 2, 6),
            shell_backed=set(dp.nodes) <= area_node_ids))

    skipped: list[str] = [
        f"diaphragm {d.name!r} -> shell mesh provides diaphragm action (skipped)"
        for d in diaphragms if d.shell_backed
    ]

    return ImportResult(
        node_tag=node_tag,
        frame_groups=frame_groups,
        area_groups=area_groups,
        restraint_groups=restraint_groups,
        load_patterns=load_patterns,
        diaphragms=diaphragms,
        has_masses=has_masses,
        skipped=skipped,
    )


def _inject_diaphragms(fem, result: ImportResult) -> None:
    """Append RIGID_DIAPHRAGM records to the FEM snapshot for non-shell-backed
    diaphragms. Master = the joint nearest the diaphragm centroid (a real,
    stiffness-connected node, so no orphan-master singularity)."""
    specs = [d for d in result.diaphragms if not d.shell_backed]
    if not specs:
        return
    import numpy as np

    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for d in specs:
        ids = [int(i) for i in fem.nodes.select(pg=d.pg).ids]
        if len(ids) < 2:
            continue
        coords = np.array([fem.nodes.coords[fem.nodes.index(i)] for i in ids])
        center = coords.mean(axis=0)
        master = ids[int(np.argmin(((coords - center) ** 2).sum(axis=1)))]
        slaves = [i for i in ids if i != master]
        fem.nodes.constraints._records.append(NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            master_node=master, slave_nodes=slaves,
            plane_normal=np.array(d.plane_normal, dtype=float),
            dofs=list(d.dofs), name=d.name,
        ))


def build_opensees(fem, model: StructuralModel, result: ImportResult,
                   *, ndm: int = 3, ndf: int = 6,
                   shell_element: str = "ASDShellT3"):
    """Wire an apeSees deck from a meshed FEM snapshot + import metadata.

    Returns the ``apeSees`` object (not yet exported) so the caller chooses
    ``ops.tcl(path)`` / ``ops.py(path)``.
    """
    # Rigid-diaphragm records must be on the snapshot before apeSees reads it.
    _inject_diaphragms(fem, result)

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
            E=mat.E, nu=mat.nu, h=sec.thickness, rho=mat.rho or 0.0)
        shell_ctor(pg=ag.pg, section=plate)

    for rg in result.restraint_groups:
        ops.fix(pg=rg.pg, dofs=tuple(rg.dofs[:ndf]))

    if result.has_masses:
        ops.mass_from_model()

    if result.load_patterns:
        ts = ops.timeSeries.Linear()
        for name in result.load_patterns:
            with ops.pattern.Plain(series=ts) as p:
                p.from_model(name)

    return ops
