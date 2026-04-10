"""
_Inspect — post-build inspection of the OpenSees model.

Accessed via ``g.opensees.inspect``.  Returns the node and element
tables that were computed during :meth:`OpenSees.build`, plus a
human-readable ``summary()`` that works before *or* after build.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .OpenSees import OpenSees


class _Inspect:
    """Post-build node / element tables and model summary."""

    def __init__(self, parent: "OpenSees") -> None:
        self._opensees = parent

    # ------------------------------------------------------------------
    # Node table (post-build)
    # ------------------------------------------------------------------

    def node_table(self) -> pd.DataFrame:
        """
        Node coordinates plus declared nodal annotations (post-build).

        The returned table is indexed by OpenSees node ID and includes:

        * ``x``, ``y``, ``z`` coordinate columns
        * ``fix_i`` boolean columns for each constrained DOF
        * ``load_i`` float columns with the cumulative nodal load per DOF
        """
        ops = self._opensees
        ops._require_built("inspect.node_table")
        df = ops._nodes_df.copy()

        for dof_idx in range(1, ops._ndf + 1):
            df[f"fix_{dof_idx}"] = False
            df[f"load_{dof_idx}"] = 0.0

        if df.empty:
            return df

        for pg_name, bc in ops._bcs.items():
            ops_ids = ops._nodes_for_pg(pg_name, bc.get("dim"))
            if not ops_ids:
                continue
            for dof_idx, is_fixed in enumerate(bc["dofs"], start=1):
                if is_fixed:
                    df.loc[ops_ids, f"fix_{dof_idx}"] = True

        for loads in ops._load_patterns.values():
            for load_def in loads:
                if load_def["type"] == "nodal":
                    ops_ids = ops._nodes_for_pg(
                        load_def["pg_name"],
                        load_def.get("dim"),
                    )
                    if not ops_ids:
                        continue
                    for dof_idx, force in enumerate(load_def["force"], start=1):
                        if force:
                            df.loc[ops_ids, f"load_{dof_idx}"] += float(force)
                elif load_def["type"] == "nodal_direct":
                    gmsh_tag = load_def["node_id"]
                    ops_id = ops._node_map.get(int(gmsh_tag))
                    if ops_id is None:
                        continue
                    for dof_idx, force in enumerate(
                        load_def["forces"][: ops._ndf], start=1
                    ):
                        if force:
                            df.loc[ops_id, f"load_{dof_idx}"] += float(force)

        return df

    # ------------------------------------------------------------------
    # Element table (post-build)
    # ------------------------------------------------------------------

    def element_table(self) -> pd.DataFrame:
        """Element connectivity table (post-build).  Indexed by OpenSees element ID."""
        ops = self._opensees
        ops._require_built("inspect.element_table")
        return ops._elements_df.copy()

    # ------------------------------------------------------------------
    # Summary (any time)
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable model description (works before and after build)."""
        ops = self._opensees
        lines = [
            f"OpenSees bridge — model: {ops._parent.name!r}",
            f"  ndm={ops._ndm}  ndf={ops._ndf}",
            "",
            f"  nDMaterials ({len(ops._nd_materials)}):",
        ]
        for i, (name, m) in enumerate(ops._nd_materials.items(), 1):
            p = "  ".join(f"{k}={v}" for k, v in m["params"].items())
            lines.append(f"    [{i}] {name!r}  →  {m['ops_type']}  {p}")

        lines += [f"  uniaxialMaterials ({len(ops._uni_materials)}):"]
        for i, (name, m) in enumerate(ops._uni_materials.items(), 1):
            p = "  ".join(f"{k}={v}" for k, v in m["params"].items())
            lines.append(f"    [{i}] {name!r}  →  {m['ops_type']}  {p}")

        lines += [f"  Sections ({len(ops._sections)}):"]
        for i, (name, s) in enumerate(ops._sections.items(), 1):
            p = "  ".join(f"{k}={v}" for k, v in s["params"].items())
            lines.append(f"    [{i}] {name!r}  →  {s['section_type']}  {p}")

        lines += [f"  GeomTransfs ({len(ops._geom_transfs)}):"]
        for i, (name, t) in enumerate(ops._geom_transfs.items(), 1):
            lines.append(f"    [{i}] {name!r}  →  {t['transf_type']}")

        lines += ["", f"  Element assignments ({len(ops._elem_assignments)}):"]
        for pg, a in ops._elem_assignments.items():
            mat_info = (
                f"mat={a['material']!r}" if a['material']
                else f"transf={a['geom_transf']!r}"
            )
            extra_str = (
                "  " + "  ".join(f"{k}={v}" for k, v in a['extra'].items())
                if a['extra'] else ""
            )
            lines.append(
                f"    PG {pg!r}  →  {a['ops_type']}  ({mat_info}){extra_str}"
            )

        lines += ["", f"  Boundary conditions ({len(ops._bcs)}):"]
        for pg, bc in ops._bcs.items():
            lines.append(f"    PG {pg!r}  →  fix {bc['dofs']}")

        lines += ["", f"  Load patterns ({len(ops._load_patterns)}):"]
        for pat, loads in ops._load_patterns.items():
            for ld in loads:
                lines.append(
                    f"    {pat!r}  PG {ld.get('pg_name', '<direct>')!r}  "
                    f"{ld['type']}  force={ld.get('force') or ld.get('forces')}"
                )

        if ops._built:
            lines += ["", "  ── built ──", f"  nodes    : {len(ops._nodes_df)}"]
            if not ops._elements_df.empty:
                by = (
                    ops._elements_df
                    .groupby(['ops_type', 'pg_name'])
                    .size()
                    .reset_index(name='n')
                )
                lines.append(f"  elements : {len(ops._elements_df)}")
                for _, r in by.iterrows():
                    lines.append(
                        f"    {r.ops_type:32s}  PG {r.pg_name!r}  n={r.n}"
                    )
        return "\n".join(lines)
