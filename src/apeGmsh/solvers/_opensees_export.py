"""
_Export — write the built OpenSees model to Tcl or openseespy scripts.

Accessed via ``g.opensees.export``.  Both exporters read exclusively
from the parent :class:`OpenSees`'s internal tables and dicts — they
perform no gmsh I/O.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._element_specs import _render_tcl, _render_py
from ._opensees_constraints import render_tie_py, render_tie_tcl

if TYPE_CHECKING:
    from .OpenSees import OpenSees


class _Export:
    """Tcl and openseespy script emitters."""

    def __init__(self, parent: "OpenSees") -> None:
        self._opensees = parent

    # ------------------------------------------------------------------
    # Tcl
    # ------------------------------------------------------------------

    def tcl(self, path: Path | str) -> "_Export":
        """Write an OpenSees Tcl input script to *path*."""
        ops = self._opensees
        ops._require_built("export.tcl")
        path = Path(path)
        lines: list[str] = []

        def hdr(title: str) -> None:
            lines.extend(["", f"# {'─'*62}", f"# {title}", f"# {'─'*62}"])

        lines += [
            "# OpenSees Tcl script",
            f"# Model : {ops._parent.name}",
            "# Source: apeGmsh / OpenSees composite",
        ]

        hdr("Model builder")
        lines.append(
            f"model BasicBuilder  -ndm {ops._ndm}  -ndf {ops._ndf}"
        )

        hdr(f"Nodes  ({len(ops._nodes_df)})")
        for ops_id, row in ops._nodes_df.iterrows():
            xyz = [row.x, row.y, row.z][: ops._ndm]
            lines.append(
                f"node {ops_id}  " + "  ".join(f"{v:.10g}" for v in xyz)
            )

        hdr(f"nDMaterials  ({len(ops._nd_materials)})")
        for name, m in ops._nd_materials.items():
            tag = ops._nd_mat_tags[name]
            params = "  ".join(str(v) for v in m["params"].values())
            lines.append(
                f"nDMaterial {m['ops_type']}  {tag}  {params}  ;# {name}"
            )

        hdr(f"uniaxialMaterials  ({len(ops._uni_materials)})")
        for name, m in ops._uni_materials.items():
            tag = ops._uni_mat_tags[name]
            params = "  ".join(str(v) for v in m["params"].values())
            lines.append(
                f"uniaxialMaterial {m['ops_type']}  {tag}  {params}  ;# {name}"
            )

        hdr(f"Sections  ({len(ops._sections)})")
        for name, s in ops._sections.items():
            tag = ops._sec_tags[name]
            params = "  ".join(str(v) for v in s["params"].values())
            lines.append(
                f"section {s['section_type']}  {tag}  {params}  ;# {name}"
            )

        hdr(f"GeomTransfs  ({len(ops._geom_transfs)})")
        for name, t in ops._geom_transfs.items():
            tag = ops._transf_tags[name]
            vecxz = t.get("vecxz")
            suffix = (
                "  " + "  ".join(str(v) for v in vecxz)
                if (vecxz and ops._ndm == 3) else ""
            )
            lines.append(
                f"geomTransf {t['transf_type']}  {tag}{suffix}  ;# {name}"
            )

        hdr(f"Elements  ({len(ops._elements_df)})")
        for ops_id, row in ops._elements_df.iterrows():
            lines.append(
                _render_tcl(
                    int(ops_id),  # type: ignore[arg-type]
                    row.ops_type, row.slots,
                    row.nodes, row.mat_tag, row.sec_tag, row.transf_tag,
                    row.extra, row.pg_name,
                )
            )

        if ops._tie_elements:
            hdr(
                f"Tied interfaces  "
                f"({len(ops._tie_elements)} ASDEmbeddedNodeElement)"
            )
            for entry in ops._tie_elements:
                lines.append(render_tie_tcl(entry))

        hdr("Single-point constraints  (fix)")
        for pg_name, bc in ops._bcs.items():
            ops_ids = ops._nodes_for_pg(pg_name, bc.get("dim"))
            dof_str = "  ".join(str(d) for d in bc["dofs"])
            lines.append(f";# PG: {pg_name!r}  —  {len(ops_ids)} nodes")
            for nid in ops_ids:
                lines.append(f"fix {nid}  {dof_str}")

        if ops._sp_records:
            hom: dict[int, list[int]] = {}
            pre: list[dict] = []
            for sp in ops._sp_records:
                ops_id = ops._node_map.get(int(sp["node_id"]))
                if ops_id is None:
                    continue
                if sp["is_homogeneous"]:
                    hom.setdefault(ops_id, []).append(sp["dof"])
                else:
                    pre.append({**sp, "ops_id": ops_id})
            if hom:
                hdr(f"Face SP — homogeneous fix  ({len(hom)} nodes)")
                for nid in sorted(hom):
                    dofs = sorted(set(hom[nid]))
                    mask = "  ".join(
                        str(1 if (d + 1) in dofs else 0)
                        for d in range(ops._ndf)
                    )
                    lines.append(f"fix {nid}  {mask}")
            if pre:
                hdr(f"Face SP — prescribed displacement  ({len(pre)} entries)")
                for sp in pre:
                    lines.append(f"sp {sp['ops_id']}  {sp['dof']}  {sp['value']:.10g}")

        if ops._mass_records:
            hdr(f"Nodal masses  ({len(ops._mass_records)} entries)")
            for mr in ops._mass_records:
                gmsh_tag = mr["node_id"]
                ops_id = ops._node_map.get(int(gmsh_tag))
                if ops_id is None:
                    continue
                vals = mr["mass"][: ops._ndf]
                v_str = "  ".join(f"{v:.10g}" for v in vals)
                lines.append(f"mass {ops_id}  {v_str}")

        hdr("Load patterns")
        for pat_idx, (pat_name, loads) in enumerate(
            ops._load_patterns.items(), start=1
        ):
            lines.append(f"pattern Plain {pat_idx} Linear {{")
            lines.append(f"    ;# pattern: {pat_name!r}")
            for ld in loads:
                if ld["type"] == "nodal":
                    ops_ids = ops._nodes_for_pg(ld["pg_name"], ld.get("dim"))
                    f_str = "  ".join(str(v) for v in ld["force"])
                    lines.append(
                        f"    ;# PG: {ld['pg_name']!r}  —  {len(ops_ids)} nodes"
                    )
                    for nid in ops_ids:
                        lines.append(f"    load {nid}  {f_str}")
                elif ld["type"] == "nodal_direct":
                    gmsh_tag = ld["node_id"]
                    ops_id = ops._node_map.get(int(gmsh_tag))
                    if ops_id is None:
                        continue
                    forces = ld["forces"][: ops._ndf]
                    f_str = "  ".join(f"{v:.10g}" for v in forces)
                    lines.append(f"    load {ops_id}  {f_str}")
                elif ld["type"] == "element_direct":
                    eid = ld["element_id"]
                    lt = ld["load_type"]
                    params = ld.get("params", {})
                    if lt == "beamUniform":
                        wy = params.get("wy", 0.0)
                        wz = params.get("wz", 0.0)
                        wx = params.get("wx", 0.0)
                        lines.append(
                            f"    eleLoad -ele {eid} -type -beamUniform "
                            f"{wy:.10g} {wz:.10g} {wx:.10g}"
                        )
                    elif lt == "surfacePressure":
                        p = params.get("p", 0.0)
                        lines.append(
                            f"    eleLoad -ele {eid} -type -surfaceLoad {p:.10g}"
                        )
                    else:
                        lines.append(
                            f"    ;# unsupported eleLoad type {lt!r} for element {eid}"
                        )
            lines.append("}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ops._log(f"export.tcl -> {path}  ({len(lines)} lines)")
        return self

    # ------------------------------------------------------------------
    # openseespy
    # ------------------------------------------------------------------

    def py(self, path: Path | str) -> "_Export":
        """Write an openseespy Python script to *path*."""
        ops = self._opensees
        ops._require_built("export.py")
        path = Path(path)
        lines: list[str] = []

        def hdr(title: str) -> None:
            lines.extend(["", f"# {'─'*62}", f"# {title}"])

        lines += [
            "# openseespy script",
            f"# Model : {ops._parent.name}",
            "# Source: apeGmsh / OpenSees composite",
            "import openseespy.opensees as ops",
        ]

        hdr("Model builder")
        lines.append(
            f"ops.model('basic', '-ndm', {ops._ndm}, '-ndf', {ops._ndf})"
        )

        hdr(f"Nodes  ({len(ops._nodes_df)})")
        for ops_id, row in ops._nodes_df.iterrows():
            xyz = [row.x, row.y, row.z][: ops._ndm]
            lines.append(
                f"ops.node({ops_id}, "
                + ", ".join(f"{v:.10g}" for v in xyz)
                + ")"
            )

        hdr(f"nDMaterials  ({len(ops._nd_materials)})")
        for name, m in ops._nd_materials.items():
            tag = ops._nd_mat_tags[name]
            params = ", ".join(str(v) for v in m["params"].values())
            lines.append(
                f"ops.nDMaterial('{m['ops_type']}', {tag}, {params})  # {name}"
            )

        hdr(f"uniaxialMaterials  ({len(ops._uni_materials)})")
        for name, m in ops._uni_materials.items():
            tag = ops._uni_mat_tags[name]
            params = ", ".join(str(v) for v in m["params"].values())
            lines.append(
                f"ops.uniaxialMaterial('{m['ops_type']}', {tag}, {params})"
                f"  # {name}"
            )

        hdr(f"Sections  ({len(ops._sections)})")
        for name, s in ops._sections.items():
            tag = ops._sec_tags[name]
            params = ", ".join(str(v) for v in s["params"].values())
            lines.append(
                f"ops.section('{s['section_type']}', {tag}, {params})  # {name}"
            )

        hdr(f"GeomTransfs  ({len(ops._geom_transfs)})")
        for name, t in ops._geom_transfs.items():
            tag = ops._transf_tags[name]
            vecxz = t.get("vecxz")
            suffix = (
                ", " + ", ".join(repr(v) for v in vecxz)
                if (vecxz and ops._ndm == 3) else ""
            )
            lines.append(
                f"ops.geomTransf('{t['transf_type']}', {tag}{suffix})  # {name}"
            )

        hdr(f"Elements  ({len(ops._elements_df)})")
        for ops_id, row in ops._elements_df.iterrows():
            lines.append(
                _render_py(
                    int(ops_id),  # type: ignore[arg-type]
                    row.ops_type, row.slots,
                    row.nodes, row.mat_tag, row.sec_tag, row.transf_tag,
                    row.extra, row.pg_name,
                )
            )

        if ops._tie_elements:
            hdr(
                f"Tied interfaces  "
                f"({len(ops._tie_elements)} ASDEmbeddedNodeElement)"
            )
            for entry in ops._tie_elements:
                lines.append(render_tie_py(entry))

        hdr("Single-point constraints")
        for pg_name, bc in ops._bcs.items():
            ops_ids = ops._nodes_for_pg(pg_name, bc.get("dim"))
            dof_str = ", ".join(str(d) for d in bc["dofs"])
            lines.append(f"# PG: {pg_name!r}  —  {len(ops_ids)} nodes")
            for nid in ops_ids:
                lines.append(f"ops.fix({nid}, {dof_str})")

        if ops._sp_records:
            # Group homogeneous records into fix() calls; emit prescribed as sp()
            hom: dict[int, list[int]] = {}   # ops_id -> sorted dof list
            pre: list[dict] = []
            for sp in ops._sp_records:
                ops_id = ops._node_map.get(int(sp["node_id"]))
                if ops_id is None:
                    continue
                if sp["is_homogeneous"]:
                    hom.setdefault(ops_id, []).append(sp["dof"])
                else:
                    pre.append({**sp, "ops_id": ops_id})
            if hom:
                hdr(f"Face SP — homogeneous fix  ({len(hom)} nodes)")
                for nid in sorted(hom):
                    dofs = sorted(set(hom[nid]))
                    mask = ", ".join(
                        str(1 if (d + 1) in dofs else 0)
                        for d in range(ops._ndf)
                    )
                    lines.append(f"ops.fix({nid}, {mask})")
            if pre:
                hdr(f"Face SP — prescribed displacement  ({len(pre)} entries)")
                for sp in pre:
                    lines.append(
                        f"ops.sp({sp['ops_id']}, {sp['dof']}, {sp['value']:.10g})"
                    )

        if ops._mass_records:
            hdr(f"Nodal masses  ({len(ops._mass_records)} entries)")
            for mr in ops._mass_records:
                gmsh_tag = mr["node_id"]
                ops_id = ops._node_map.get(int(gmsh_tag))
                if ops_id is None:
                    continue
                vals = mr["mass"][: ops._ndf]
                v_str = ", ".join(f"{v:.10g}" for v in vals)
                lines.append(f"ops.mass({ops_id}, {v_str})")

        hdr("Load patterns")
        for pat_idx, (pat_name, loads) in enumerate(
            ops._load_patterns.items(), start=1
        ):
            lines.append(
                f"ops.pattern('Plain', {pat_idx}, 'Linear')  # {pat_name!r}"
            )
            for ld in loads:
                if ld["type"] == "nodal":
                    ops_ids = ops._nodes_for_pg(ld["pg_name"], ld.get("dim"))
                    f_str = ", ".join(str(v) for v in ld["force"])
                    lines.append(
                        f"# PG: {ld['pg_name']!r}  —  {len(ops_ids)} nodes"
                    )
                    for nid in ops_ids:
                        lines.append(f"ops.load({nid}, {f_str})")
                elif ld["type"] == "nodal_direct":
                    gmsh_tag = ld["node_id"]
                    ops_id = ops._node_map.get(int(gmsh_tag))
                    if ops_id is None:
                        continue
                    forces = ld["forces"][: ops._ndf]
                    f_str = ", ".join(f"{v:.10g}" for v in forces)
                    lines.append(f"ops.load({ops_id}, {f_str})")
                elif ld["type"] == "element_direct":
                    eid = ld["element_id"]
                    lt = ld["load_type"]
                    params = ld.get("params", {})
                    if lt == "beamUniform":
                        wy = params.get("wy", 0.0)
                        wz = params.get("wz", 0.0)
                        wx = params.get("wx", 0.0)
                        lines.append(
                            f"ops.eleLoad('-ele', {eid}, '-type', '-beamUniform', "
                            f"{wy:.10g}, {wz:.10g}, {wx:.10g})"
                        )
                    elif lt == "surfacePressure":
                        p = params.get("p", 0.0)
                        lines.append(
                            f"ops.eleLoad('-ele', {eid}, '-type', '-surfaceLoad', {p:.10g})"
                        )
                    else:
                        lines.append(
                            f"# unsupported eleLoad type {lt!r} for element {eid}"
                        )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ops._log(f"export.py -> {path}  ({len(lines)} lines)")
        return self
