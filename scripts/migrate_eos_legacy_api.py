"""Adapt EOS notebooks to the current apeGmsh API.

In-place rewrites for these breakages:

1. ``Results.from_fem(fem, steps=..., include_pgs=..., name=...)``
   was removed in the Phase 0-8 Results rebuild. Replace each call
   site with inline ``NativeWriter`` + ``Results.from_native`` code.
   Vector fields (shape ``(N, 3)``) split into per-axis scalars.

2. ``g.mesh.partitioning.renumber_mesh(...)`` was renamed to
   ``g.mesh.partitioning.renumber(...)``.

3. ``Results(node_coords=..., ...)`` direct constructor was removed.
   Replace with ``Results.from_native(path, fem=fem)`` after the
   migrated NativeWriter block.

4. ``example_column_nodeToSurface.ipynb``'s missing
   ``load_factor_history``/``displacement_results``/
   ``reactions_results`` initialisation — patch in the missing
   defaultdict-style init before the analysis loop.

Idempotent — looks for ``# --- LEGACY-API-MIGRATED ---`` and skips.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import nbformat

MARKER = "# --- LEGACY-API-MIGRATED ---"

# ---------------------------------------------------------------------------
# 1. Results.from_fem  ->  inline NativeWriter + Results.from_native
# ---------------------------------------------------------------------------

_FROM_FEM_START = re.compile(r"\bResults\.from_fem\s*\(", re.MULTILINE)


def _balanced_call_end(src: str, open_idx: int) -> int:
    """Given the index of the '(' opening a call, return the index AFTER ')'."""
    depth = 1
    i = open_idx + 1
    while i < len(src) and depth > 0:
        ch = src[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    return i  # one past the closing ')'


def _split_top_level_args(args_src: str) -> list[str]:
    """Split a Python call argument body into top-level args (commas not
    nested in (), [], {} or strings). Quotes assumed simple (no triple)."""
    parts: list[str] = []
    cur = []
    depth = 0
    in_str = None
    i = 0
    while i < len(args_src):
        ch = args_src[i]
        if in_str:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(args_src):
                cur.append(args_src[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ("'", '"'):
            in_str = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
    return parts


def _parse_kwargs(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """Split call args into (positional, kwargs)."""
    pos = []
    kw: dict[str, str] = {}
    for a in args:
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", a, re.DOTALL)
        if m:
            kw[m.group(1)] = m.group(2).strip()
        else:
            pos.append(a)
    return pos, kw


_REPLACEMENT_TEMPLATE_STEPS = '''\
{indent}# --- LEGACY-API-MIGRATED ---
{indent}# Materialize ``{steps_var}`` into ``(T, N)`` arrays for ``NativeWriter``.
{indent}# Vector fields (shape ``(N, 3)``) split into per-axis scalar components.
{indent}_legacy_components = {{}}
{indent}for _legacy_cname, _legacy_cval0 in {steps_var}[0]["point_data"].items():
{indent}    _legacy_arr0 = np.asarray(_legacy_cval0)
{indent}    if _legacy_arr0.ndim == 2 and _legacy_arr0.shape[1] in (2, 3):
{indent}        for _legacy_i, _legacy_ax in enumerate(["x", "y", "z"][: _legacy_arr0.shape[1]]):
{indent}            _legacy_components[f"{{_legacy_cname}}_{{_legacy_ax}}"] = np.stack(
{indent}                [np.asarray(_s["point_data"][_legacy_cname])[:, _legacy_i]
{indent}                 for _s in {steps_var}], axis=0,
{indent}            )
{indent}    else:
{indent}        _legacy_components[_legacy_cname] = np.stack(
{indent}            [np.asarray(_s["point_data"][_legacy_cname]) for _s in {steps_var}],
{indent}            axis=0,
{indent}        )
{indent}_legacy_time = np.array([_s["time"] for _s in {steps_var}], dtype=float)
{indent}# Determine node count from the data, then pick node_ids that match.
{indent}_legacy_first = next(iter({steps_var}[0]["point_data"].values()))
{indent}_legacy_N = int(np.asarray(_legacy_first).shape[0])
{indent}if {include_pgs_expr}:
{indent}    _legacy_node_ids = np.asarray(
{indent}        {fem_var}.nodes.get_ids(pg={pg_first_expr}), dtype=np.int64,
{indent}    )
{indent}    if _legacy_node_ids.size != _legacy_N:
{indent}        # Falls back to all nodes if the pg cardinality doesn't match
{indent}        # the per-step data (covers cases where the user collected
{indent}        # disp on a different subset than the legacy ``include_pgs``).
{indent}        _legacy_node_ids = np.asarray({fem_var}.nodes.ids, dtype=np.int64)
{indent}else:
{indent}    _legacy_node_ids = np.asarray({fem_var}.nodes.ids, dtype=np.int64)
{indent}if _legacy_node_ids.size != _legacy_N:
{indent}    raise RuntimeError(
{indent}        f"node_ids has {{_legacy_node_ids.size}} entries but step data "
{indent}        f"has {{_legacy_N}} — adjust the source of node_ids in this cell."
{indent}    )
{indent}from pathlib import Path as _LegacyPath
{indent}from apeGmsh.results.writers import NativeWriter as _LegacyNativeWriter
{indent}_legacy_path = _LegacyPath(f"{{{name_expr}}}_legacy.h5")
{indent}if _legacy_path.exists():
{indent}    _legacy_path.unlink()
{indent}with _LegacyNativeWriter(_legacy_path) as _legacy_nw:
{indent}    _legacy_nw.open(fem={fem_var})
{indent}    _legacy_sid = _legacy_nw.begin_stage(name={name_expr}, kind="static", time=_legacy_time)
{indent}    _legacy_nw.write_nodes(
{indent}        _legacy_sid, "partition_0",
{indent}        node_ids=_legacy_node_ids, components=_legacy_components,
{indent}    )
{indent}    _legacy_nw.end_stage()
{indent}{lhs}Results.from_native(_legacy_path, fem={fem_var})'''


# Single-snapshot variant: ``Results.from_fem(fem, point_data={{...}}, ...)``
# becomes a (T=1, N) NativeWriter write.
_REPLACEMENT_TEMPLATE_POINT_DATA = '''\
{indent}# --- LEGACY-API-MIGRATED ---
{indent}# Single-snapshot ``point_data`` dict -> (T=1, N) NativeWriter write.
{indent}# Vector fields (shape ``(N, 3)``) split into per-axis scalar components.
{indent}_legacy_pd = {point_data_expr}
{indent}_legacy_components = {{}}
{indent}for _legacy_cname, _legacy_cval in _legacy_pd.items():
{indent}    _legacy_arr = np.asarray(_legacy_cval)
{indent}    if _legacy_arr.ndim == 2 and _legacy_arr.shape[1] in (2, 3):
{indent}        for _legacy_i, _legacy_ax in enumerate(["x", "y", "z"][: _legacy_arr.shape[1]]):
{indent}            _legacy_components[f"{{_legacy_cname}}_{{_legacy_ax}}"] = (
{indent}                _legacy_arr[:, _legacy_i].reshape(1, -1)
{indent}            )
{indent}    else:
{indent}        _legacy_components[_legacy_cname] = _legacy_arr.reshape(1, -1)
{indent}_legacy_time = np.array([1.0], dtype=float)
{indent}_legacy_first = next(iter(_legacy_pd.values()))
{indent}_legacy_N = int(np.asarray(_legacy_first).shape[0])
{indent}if {include_pgs_expr}:
{indent}    _legacy_node_ids = np.asarray(
{indent}        {fem_var}.nodes.get_ids(pg={pg_first_expr}), dtype=np.int64,
{indent}    )
{indent}    if _legacy_node_ids.size != _legacy_N:
{indent}        _legacy_node_ids = np.asarray({fem_var}.nodes.ids, dtype=np.int64)
{indent}else:
{indent}    _legacy_node_ids = np.asarray({fem_var}.nodes.ids, dtype=np.int64)
{indent}if _legacy_node_ids.size != _legacy_N:
{indent}    raise RuntimeError(
{indent}        f"node_ids has {{_legacy_node_ids.size}} entries but data "
{indent}        f"has {{_legacy_N}}."
{indent}    )
{indent}from pathlib import Path as _LegacyPath
{indent}from apeGmsh.results.writers import NativeWriter as _LegacyNativeWriter
{indent}_legacy_path = _LegacyPath(f"{{{name_expr}}}_legacy.h5")
{indent}if _legacy_path.exists():
{indent}    _legacy_path.unlink()
{indent}with _LegacyNativeWriter(_legacy_path) as _legacy_nw:
{indent}    _legacy_nw.open(fem={fem_var})
{indent}    _legacy_sid = _legacy_nw.begin_stage(name={name_expr}, kind="static", time=_legacy_time)
{indent}    _legacy_nw.write_nodes(
{indent}        _legacy_sid, "partition_0",
{indent}        node_ids=_legacy_node_ids, components=_legacy_components,
{indent}    )
{indent}    _legacy_nw.end_stage()
{indent}{lhs}Results.from_native(_legacy_path, fem={fem_var})'''


def _migrate_results_from_fem(src: str) -> tuple[str, int]:
    """Replace each ``Results.from_fem(...)`` call in ``src`` with an inline
    NativeWriter+from_native block. Returns (new_src, n_replacements)."""
    out = src
    n = 0
    while True:
        m = _FROM_FEM_START.search(out)
        if m is None:
            break
        # Find the start of the call line (for indentation + LHS).
        line_start = out.rfind("\n", 0, m.start()) + 1
        line_prefix = out[line_start : m.start()]
        # Find LHS like ``foo = `` if present.
        lhs_match = re.match(r"^(\s*)([A-Za-z_][\w\.\[\]]*\s*=\s*)?$", line_prefix)
        if lhs_match is None:
            # Unexpected — bail out without replacing.
            break
        indent = lhs_match.group(1) or ""
        lhs = lhs_match.group(2) or ""

        open_paren = m.end() - 1
        end = _balanced_call_end(out, open_paren)
        args_src = out[open_paren + 1 : end - 1]
        args = _split_top_level_args(args_src)
        pos, kw = _parse_kwargs(args)
        fem_var = pos[0] if pos else kw.get("fem", "fem")
        include_pgs_expr = kw.get("include_pgs", "None")
        name_expr = kw.get("name", '"analysis"')
        if include_pgs_expr.startswith("[") and include_pgs_expr.endswith("]"):
            inner = include_pgs_expr[1:-1].strip()
            first = _split_top_level_args(inner)[0] if inner else "None"
            pg_first_expr = first
        else:
            pg_first_expr = (
                f"({include_pgs_expr})[0] if {include_pgs_expr} else None"
            )

        if "steps" in kw:
            steps_var = kw["steps"]
            replacement = _REPLACEMENT_TEMPLATE_STEPS.format(
                indent=indent, lhs=lhs, fem_var=fem_var,
                steps_var=steps_var,
                include_pgs_expr=include_pgs_expr,
                pg_first_expr=pg_first_expr, name_expr=name_expr,
            )
        elif "point_data" in kw:
            point_data_expr = kw["point_data"]
            replacement = _REPLACEMENT_TEMPLATE_POINT_DATA.format(
                indent=indent, lhs=lhs, fem_var=fem_var,
                point_data_expr=point_data_expr,
                include_pgs_expr=include_pgs_expr,
                pg_first_expr=pg_first_expr, name_expr=name_expr,
            )
        else:
            # Unknown signature — bail out without replacing this call.
            break

        out = out[:line_start] + replacement + out[end:]
        n += 1
    return out, n


# ---------------------------------------------------------------------------
# 2. renumber_mesh -> renumber
# ---------------------------------------------------------------------------


def _migrate_renumber_mesh(src: str) -> tuple[str, int]:
    new = re.sub(
        r"\bpartitioning\.renumber_mesh\(",
        "partitioning.renumber(",
        src,
    )
    if new != src:
        return new, src.count("partitioning.renumber_mesh(")
    return src, 0


# ---------------------------------------------------------------------------
# Pre-rebuild FEMData attribute names: ``fem.node_ids`` etc. were renamed
# to ``fem.nodes.ids`` / ``fem.elements.connectivity`` in the v2 broker.
# ---------------------------------------------------------------------------

_FEM_RENAMES = [
    (re.compile(r"\bfem\.node_ids\b"),         "fem.nodes.ids"),
    (re.compile(r"\bfem\.node_coords\b"),      "fem.nodes.coords"),
    (re.compile(r"\bfem\.connectivity\b"),     "fem.elements.connectivity"),
    (re.compile(r"\bfem\.element_ids\b"),      "fem.elements.ids"),
    (re.compile(r"\bfem\.n_nodes\b"),          "fem.info.n_nodes"),
    (re.compile(r"\bfem\.n_elems\b"),          "fem.info.n_elems"),
]


# ``fem.physical.get_elements(dim, pg)`` returned a dict {'ids', 'connectivity'};
# replaced by ``fem.elements.resolve(pg=pg, dim=dim)`` returning ``(ids, conn)``.
_FEM_PHYSICAL_GET_ELEMENTS = re.compile(
    r"\bfem\.physical\.get_elements\(\s*(\d+)\s*,\s*([^)]+)\)",
)


def _migrate_fem_physical_get_elements(src: str) -> tuple[str, int]:
    """Replace ``fem.physical.get_elements(dim, pg)`` with a small block that
    builds an equivalent dict via ``fem.elements.resolve``.

    The original returned ``{'ids': ..., 'connectivity': ...}``. To preserve
    that subscriptable pattern (``inner['connectivity']``), we wrap the
    resolve return into a dict literal in-place.
    """
    def _sub(m: re.Match) -> str:
        dim = m.group(1)
        pg_expr = m.group(2).strip()
        return (
            "(lambda _ids_conn: {'ids': _ids_conn[0], 'connectivity': _ids_conn[1]})"
            f"(fem.elements.resolve(pg={pg_expr}, dim={dim}))"
        )
    new, n = _FEM_PHYSICAL_GET_ELEMENTS.subn(_sub, src)
    return new, n


def _migrate_fem_attrs(src: str) -> tuple[str, int]:
    new = src
    n = 0
    for pat, repl in _FEM_RENAMES:
        new, k = pat.subn(repl, new)
        n += k
    return new, n


# ---------------------------------------------------------------------------
# 3. Results(node_coords=...) direct constructor  -> from_native
# ---------------------------------------------------------------------------

# This one is column_v6-specific. We skip it generically here and hand-edit
# the offending notebook directly.


# ---------------------------------------------------------------------------
# 4. column_nodeToSurface NameError init bug
# ---------------------------------------------------------------------------


def _patch_column_v1_init(nb) -> int:
    """Insert the missing init for load_factor_history / displacement_results /
    reactions_results immediately before the first cell that appends to them.
    Returns 1 if patched, 0 if no change needed."""
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if (
            "load_factor_history.append" in c.source
            and "load_factor_history = []" not in "\n".join(
                cc.source for cc in nb.cells[: i + 1] if cc.cell_type == "code"
            )
        ):
            init_block = (
                f"{MARKER}\n"
                "# Initialise per-step result containers used by the analysis loop\n"
                "# below. (Originally absent from the notebook — added here so it\n"
                "# runs end-to-end against the current apeGmsh API.)\n"
                "load_factor_history: list[float] = []\n"
                "displacement_results = {int(_nid): [] for _nid in monitored_ids}\n"
                "reactions_results    = {int(_nid): [] for _nid in monitored_ids}\n"
            )
            nb.cells.insert(i, nbformat.v4.new_code_cell(init_block))
            return 1
    return 0


# ---------------------------------------------------------------------------
# Top-level: run all migrations on a notebook
# ---------------------------------------------------------------------------


def migrate(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    if any(c.cell_type == "code" and MARKER in c.source for c in nb.cells):
        return f"[skip] {nb_path.name}: already migrated"

    n_from_fem = 0
    n_renumber = 0
    n_fem_attrs = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        new_src, k1 = _migrate_results_from_fem(c.source)
        new_src, k2 = _migrate_renumber_mesh(new_src)
        new_src, k3 = _migrate_fem_attrs(new_src)
        new_src, k4 = _migrate_fem_physical_get_elements(new_src)
        if k1 + k2 + k3 + k4 > 0:
            c.source = new_src
            n_from_fem += k1
            n_renumber += k2
            n_fem_attrs += k3 + k4

    n_init = 0
    if "column_nodeToSurface.ipynb" in nb_path.name and not any(
        v in nb_path.name for v in ("_v2", "_v3", "_v4", "_v5", "_v6")
    ):
        n_init = _patch_column_v1_init(nb)

    if n_from_fem + n_renumber + n_fem_attrs + n_init == 0:
        return f"[skip] {nb_path.name}: no known breakages"

    nbformat.write(nb, nb_path)
    return (
        f"[migrated] {nb_path.name}: from_fem={n_from_fem} "
        f"renumber={n_renumber} fem_attrs={n_fem_attrs} init={n_init}"
    )


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    eos = Path("examples/EOS Examples")
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        targets = sorted(eos.glob("*.ipynb"))
    else:
        targets = [eos / a for a in sys.argv[1:]]
    for p in targets:
        print(migrate(p))
