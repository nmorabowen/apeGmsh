"""Ladruno (.ladruno) reader + schema-v1 validator + canonical normalizers.

This module is the **executable form of the schema spec** (ladruno_schema_v1.md).
Three jobs:

1. ``LadrunoReader`` -- walk a .ladruno HDF5 file per schema v1.
2. ``validate`` -- assert every structural invariant the spec promises (L2 of the test
   plan). Returns a list of problems; empty == conformant.
3. ``normalize_nodal`` / ``normalize_element`` -- flatten results to canonical
   ``{(locator...): value}`` dicts so the L1 parity gate can diff a .ladruno against a
   legacy .mpco *by information, not layout*.

Pure h5py + numpy. No OpenSees, no build. When the C++ recorder emits its first file,
this validates it immediately.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import h5py
import numpy as np

import ladruno_basis

GENERATOR = "Ladruno"
FORMAT_VERSION = 1
# WP-232a: INFO/FORMAT_MINOR_VERSION 2 = minor 1 (schema §2a) + the MPM particle group
# (MODEL/PARTICLES + RESULTS/ON_PARTICLES, Ladruno_implementation/232a_mpm_contracts.md §2).
# Additive: FORMAT_VERSION stays 1. A file with a particle group must write minor >= 2.
FORMAT_MINOR_VERSION_PARTICLES = 2

# COLUMN_MAP/LEVELS path codes (schema §7.2)
LEVEL_ELEMENT, LEVEL_GAUSS, LEVEL_SECTION, LEVEL_FIBER, LEVEL_MATERIAL = 0, 1, 2, 3, 4


def _attr(obj: h5py.HLObject, name: str) -> Any:
    v = obj.attrs[name]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "S":
            decoded = [x.decode("utf-8") for x in v.reshape(-1)]
            return decoded[0] if v.size == 1 else decoded
        # The MPCO-style h5 wrapper writes "scalar" int/double attributes as
        # 1-element 1-D arrays. Squeeze those back to a Python scalar so callers
        # can int()/float() them; keep genuine vectors (size>1) as arrays.
        if v.size == 1:
            return v.reshape(-1)[0].item()
    return v


def _decode_lines(raw: Any) -> list[str]:
    """COMP_NAMES is newline-separated, one line per COLUMN_MAP block."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return [ln for ln in str(raw).split("\n")]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class LadrunoReader:
    """Minimal schema-v1 reader. Open with a path; use as a context manager."""

    def __init__(self, path: str):
        self.path = path
        self.f = h5py.File(path, "r")

    def __enter__(self) -> "LadrunoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.f.close()

    # -- INFO --
    def info(self) -> dict:
        g = self.f["INFO"]
        out = {k: _attr(g, k) for k in g.attrs}
        return out

    def stored_precision(self) -> str:
        """On-disk precision of the per-step result DATA: "f64" (lossless, the
        default — and what older files implicitly are) or "f32" (opt-in
        `-precision f32` lossy mode). Consumers MUST NOT diff an "f32" file
        against an f64 oracle at 1e-12 — use a bounded-error check instead."""
        return str(self.info().get("STORED_PRECISION", "f64"))

    # -- stages --
    def stages(self) -> list[str]:
        return [k for k in self.f if k.startswith("MODEL_STAGE")]

    def stage_attrs(self, stage: str) -> dict:
        g = self.f[stage]
        return {k: _attr(g, k) for k in g.attrs}

    # -- model --
    def nodes(self, stage: str) -> tuple[np.ndarray, np.ndarray]:
        g = self.f[f"{stage}/MODEL/NODES"]
        return g["ID"][...], g["COORDINATES"][...]

    def element_groups(self, stage: str) -> dict[str, dict]:
        """name -> {connectivity, topology, family, order, param_domain, rational,
        num_ctrl, ndir, num_gp, gp_param, gp_weight, ctrl_weight?, global_gp_coords?}.

        QUADRATURE-TOLERANT: a group whose integration rule is not tabulated emits
        no QUADRATURE child -- then gp_param/gp_weight/num_gp/ndir are None (the
        caller warns, it is not an error). NDIR (schema-v1 D4) is authoritative for
        the parametric dimension, decoupled from len(ORDER); it falls back to the
        GP_PARAM column count for older files that predate the attribute.
        GLOBAL_GP_COORDS (D2) is optional, stored 2-D [nElem x (NUM_GP*ndim)]."""
        out: dict[str, dict] = {}
        base = self.f[f"{stage}/MODEL/ELEMENTS"]
        for name in base:
            grp = base[name]
            # Schema §3.4 (WP-226): a class whose elements carry different node
            # counts (RBE3/RBE2 footprints) is CSR — ID + CONNECTIVITY_OFFSETS +
            # CONNECTIVITY_NODES, NUM_CTRL=-1, no rectangular CONNECTIVITY.
            # ``connectivity`` is then None and ``rows`` lists (tag, nodes) pairs.
            csr = _attr(grp, "CONNECTIVITY_LAYOUT") == "csr" if "CONNECTIVITY_LAYOUT" in grp.attrs else False
            if csr:
                ids = np.asarray(grp["ID"][...]).reshape(-1)
                offsets = np.asarray(grp["CONNECTIVITY_OFFSETS"][...]).reshape(-1)
                nodes = np.asarray(grp["CONNECTIVITY_NODES"][...]).reshape(-1)
                conn = None
            else:
                conn = grp["CONNECTIVITY"][...]
            d = {
                "layout": "csr" if csr else "rect",
                "connectivity": conn,
                "topology": _attr(grp, "TOPOLOGY"),
                "family": _attr(grp, "FAMILY"),
                "order": tuple(int(x) for x in np.atleast_1d(_attr(grp, "ORDER")))
                if "ORDER" in grp.attrs else (),
                "param_domain": _attr(grp, "PARAM_DOMAIN"),
                "rational": int(_attr(grp, "RATIONAL")),
                "num_ctrl": int(_attr(grp, "NUM_CTRL")),
            }
            if csr:
                d["ids"] = ids
                d["offsets"] = offsets
                d["nodes"] = nodes
                d["rows"] = [
                    (int(ids[e]), nodes[int(offsets[e]):int(offsets[e + 1])])
                    for e in range(ids.shape[0])
                ] if offsets.shape[0] == ids.shape[0] + 1 else []
            if "QUADRATURE/GP_PARAM" in grp:
                gp_param = grp["QUADRATURE/GP_PARAM"][...]
                d["gp_param"] = gp_param
                d["num_gp"] = (int(_attr(grp, "NUM_GP")) if "NUM_GP" in grp.attrs
                               else gp_param.shape[0])
                d["ndir"] = (int(_attr(grp, "NDIR")) if "NDIR" in grp.attrs
                             else gp_param.shape[1])
                d["gp_weight"] = (grp["QUADRATURE/GP_WEIGHT"][...]
                                  if "QUADRATURE/GP_WEIGHT" in grp else None)
            else:
                d["gp_param"] = None
                d["num_gp"] = None
                d["ndir"] = None
                d["gp_weight"] = None
            if "CTRL_WEIGHT" in grp:
                d["ctrl_weight"] = grp["CTRL_WEIGHT"][...]
            if "GLOBAL_GP_COORDS" in grp:
                d["global_gp_coords"] = grp["GLOBAL_GP_COORDS"][...]
            out[name] = d
        return out

    def local_axes(self, stage: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        path = f"{stage}/MODEL/LOCAL_AXES"
        if path not in self.f:
            return out
        for name in self.f[path]:
            grp = self.f[path][name]
            out[name] = {"id": grp["ID"][...], "frame": grp["FRAME"][...]}
        return out

    def envelopes(self, stage: str) -> dict[str, dict]:
        """family -> {name -> {id, min, max, absmax, arg_step[, components, columns]}}
        for RESULTS/ENVELOPES (ADR D7). Empty when the recorder ran in time-series
        (non-envelope) mode.

        Node/domain envelopes are flat (ENVELOPES/<fam>/<name>); element envelopes
        nest under the result display group (ENVELOPES/ON_ELEMENTS/<display>/<bucket>),
        so leaves are found by walking to groups that carry the MIN accumulator. The
        returned name is the family-relative path ("<display>/<bucket>" for elements).
        Element leaves also carry the structured per-column COLUMN_MAP (decoded into
        `columns`); the result-level COMPONENTS attr is empty for element results."""
        out: dict[str, dict] = {}
        path = f"{stage}/RESULTS/ENVELOPES"
        if path not in self.f:
            return out
        for fam in self.f[path]:
            fam_grp = self.f[path][fam]
            out[fam] = {}
            for name, g in _iter_envelope_leaves(fam_grp):
                d = {
                    "id": g["ID"][...], "min": g["MIN"][...], "max": g["MAX"][...],
                    "absmax": g["ABSMAX"][...], "arg_step": g["ARG_STEP"][...],
                }
                if "COMPONENTS" in g.attrs:
                    d["components"] = [c for c in _attr(g, "COMPONENTS").split(",") if c]
                if "COLUMN_MAP" in g:
                    d["columns"] = decode_columns(g["COLUMN_MAP"])
                out[fam][name] = d
        return out

    # -- MPM particles (WP-232a, contract 2) --
    def particle_stages(self) -> list[str]:
        """Stages that carry a particle group, in numeric stage-id order."""
        out = [s for s in self.stages() if f"{s}/MODEL/PARTICLES" in self.f]
        return sorted(out, key=_stage_id)

    def particle_output(self, stage: str, field: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(ids [n], values [n x nComp]) of output ``k`` of ``field`` (232a §2.5)."""
        g = self.f[f"{stage}/RESULTS/ON_PARTICLES"]
        off = np.asarray(g["OFFSETS"][...]).reshape(-1)
        a, b = int(off[k]), int(off[k + 1])
        return np.asarray(g["ID"][a:b]), np.asarray(g[field]["DATA"][a:b])

    def section_assignments(self, stage: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        path = f"{stage}/MODEL/SECTION_ASSIGNMENTS"
        if path not in self.f:
            return out
        for name in self.f[path]:
            grp = self.f[path][name]
            d = {"kind": _attr(grp, "KIND"), "assignment": grp["ASSIGNMENT"][...]}
            if "FIBER_DATA" in grp:
                d["fiber_data"] = grp["FIBER_DATA"][...]
                d["fiber_materials"] = grp["FIBER_MATERIALS"][...]
            out[name] = d
        return out


# ---------------------------------------------------------------------------
# COLUMN_MAP decode (schema §7.2)
# ---------------------------------------------------------------------------


def decode_columns(cmap_grp: h5py.Group) -> list[dict]:
    """Expand COLUMN_MAP into a flat per-column list:
    [{col, gauss_id, section_tag, fiber_id, comp}].

    A block i covers MULTIPLICITY[i] consecutive items (incrementing FIBER_ID, or
    GAUSS_ID if no fiber level) each with NUM_COMP[i] components named in COMP_NAMES[i].
    """
    levels = cmap_grp["LEVELS"][...]
    # The real recorder writes these per-block scalar arrays as 2-D [k x 1]
    # (createAndWrite(vec, k, 1)); make_synthetic writes them 1-D [k]. Coerce to 1-D
    # so int(...) works under numpy 2.x for both. (LEVELS may legitimately be 2-D
    # [k x ncodes]; it is consumed via np.atleast_1d below, so leave it as-is.)
    gauss = np.asarray(cmap_grp["GAUSS_ID"][...]).reshape(-1)
    sect = np.asarray(cmap_grp["SECTION_TAG"][...]).reshape(-1)
    fiber = np.asarray(cmap_grp["FIBER_ID"][...]).reshape(-1)
    ncomp = np.asarray(cmap_grp["NUM_COMP"][...]).reshape(-1)
    mult = np.asarray(cmap_grp["MULTIPLICITY"][...]).reshape(-1)
    names = _decode_lines(_attr(cmap_grp, "COMP_NAMES"))

    cols: list[dict] = []
    col = 0
    for i in range(len(ncomp)):
        comp_names = [c for c in names[i].split(",") if c]
        has_fiber = LEVEL_FIBER in np.atleast_1d(levels[i]) if levels.ndim > 1 else (
            int(levels[i]) == LEVEL_FIBER
        )
        for m in range(int(mult[i])):
            fib = int(fiber[i]) + m if int(fiber[i]) >= 0 else -1
            gp = int(gauss[i]) + (m if (not _has_fiber(levels, i) and int(gauss[i]) >= 0) else 0)
            for c in range(int(ncomp[i])):
                cols.append(
                    {
                        "col": col,
                        "gauss_id": gp,
                        "section_tag": int(sect[i]),
                        "fiber_id": fib,
                        "comp": comp_names[c] if c < len(comp_names) else f"c{c}",
                    }
                )
                col += 1
    return cols


def _has_fiber(levels: np.ndarray, i: int) -> bool:
    row = levels[i]
    return LEVEL_FIBER in np.atleast_1d(row)


def _iter_envelope_leaves(grp, prefix: str = ""):
    """Yield (family_relative_name, leaf_group) for every envelope leaf under a
    family group. A leaf is a group that carries the MIN accumulator; element
    envelopes nest one level (ENVELOPES/ON_ELEMENTS/<display>/<bucket>) while
    node/domain envelopes are flat, so recurse through intermediate groups."""
    for name in grp:
        child = grp[name]
        if not isinstance(child, h5py.Group):
            continue
        rel = f"{prefix}{name}"
        if "MIN" in child:
            yield rel, child
        else:
            yield from _iter_envelope_leaves(child, f"{rel}/")


def iter_step_slices(grp):
    """Yield (step_id:int, arr2d[nIds×nComp]) for a result/bucket group, handling
    BOTH layouts transparently:

    * chunked (schema D3, .ladruno): DATA is a single [T×nIds×nComp] dataset with
      a sibling STEP[T] axis giving each slab's commit/step id;
    * legacy (frozen .mpco): DATA is a group of per-step STEP_<k> datasets, each
      carrying a scalar STEP attribute.

    This lets every reader/checker diff a chunked .ladruno against a per-step
    .mpco with one code path."""
    data = grp["DATA"]
    if isinstance(data, h5py.Dataset):
        full = np.asarray(data[...])
        steps = (np.asarray(grp["STEP"][...]).reshape(-1) if "STEP" in grp
                 else np.arange(full.shape[0]))
        for t in range(full.shape[0]):
            yield int(steps[t]), np.atleast_2d(full[t])
    else:
        for sname in data:
            ds = data[sname]
            yield int(_attr(ds, "STEP")), np.atleast_2d(ds[...])


def _check_data_shape(grp, n_rows: int, n_cols: int, err, ctx: str) -> None:
    """Validate the DATA payload shape for both layouts."""
    data = grp["DATA"]
    if isinstance(data, h5py.Dataset):
        if data.ndim != 3 or tuple(data.shape[1:]) != (n_rows, n_cols):
            err(f"{ctx}: DATA {data.shape} != (T, {n_rows}, {n_cols})")
        T = data.shape[0]
        for ax in ("STEP", "TIME"):
            if ax in grp and np.asarray(grp[ax][...]).reshape(-1).shape[0] != T:
                err(f"{ctx}: {ax} length != T={T}")
    else:
        for sn in data:
            item = data[sn]
            if isinstance(item, h5py.Group):
                # modal layout (modesOfVibration): DATA/STEP_<step>/MODE_<k>, each
                # MODE_<k> a [n_rows x n_cols] eigenvector dataset (mirrors frozen MPCO).
                for mn in item:
                    md = item[mn]
                    if isinstance(md, h5py.Dataset) and md.shape != (n_rows, n_cols):
                        err(f"{ctx}/{sn}/{mn}: {md.shape} != ({n_rows}, {n_cols})")
            elif item.shape != (n_rows, n_cols):
                err(f"{ctx}/{sn}: {item.shape} != ({n_rows}, {n_cols})")


# ---------------------------------------------------------------------------
# Canonical normalizers (for the L1 parity diff)
# ---------------------------------------------------------------------------


def normalize_nodal(reader: LadrunoReader, stage: str) -> dict[tuple, float]:
    """{(node_tag, component_name, step): value} over all ON_NODES results."""
    out: dict[tuple, float] = {}
    base_path = f"{stage}/RESULTS/ON_NODES"
    if base_path not in reader.f:
        return out
    base = reader.f[base_path]
    for rname in base:
        grp = base[rname]
        ids = np.asarray(grp["ID"][...]).reshape(-1)
        comps = [c for c in _attr(grp, "COMPONENTS").split(",") if c]
        for k, arr in iter_step_slices(grp):
            for r, tag in enumerate(ids):
                for c, cn in enumerate(comps):
                    out[(int(tag), f"{rname}:{cn}", k)] = float(arr[r, c])
    return out


def normalize_element(reader: LadrunoReader, stage: str) -> dict[tuple, float]:
    """{(elem_tag, gauss_id, section_tag, fiber_id, component, step): value}."""
    out: dict[tuple, float] = {}
    base_path = f"{stage}/RESULTS/ON_ELEMENTS"
    if base_path not in reader.f:
        return out
    base = reader.f[base_path]
    for rname in base:
        rgrp = base[rname]
        for bname in rgrp:
            bucket = rgrp[bname]
            if "COLUMN_MAP" not in bucket:  # e.g. LOCAL_AXES time series -> skip here
                continue
            cols = decode_columns(bucket["COLUMN_MAP"])
            ids = np.asarray(bucket["ID"][...]).reshape(-1)
            for k, arr in iter_step_slices(bucket):
                for r, tag in enumerate(ids):
                    for cd in cols:
                        out[
                            (
                                int(tag),
                                cd["gauss_id"],
                                cd["section_tag"],
                                cd["fiber_id"],
                                f"{rname}:{cd['comp']}",
                                k,
                            )
                        ] = float(arr[r, cd["col"]])
    return out


# ---------------------------------------------------------------------------
# Validator (L2) -- the executable spec
# ---------------------------------------------------------------------------


def validate(path: str) -> list[str]:
    """Return a list of schema-v1 violations. Empty list == conformant file."""
    problems: list[str] = []

    def err(msg: str) -> None:
        problems.append(msg)

    with LadrunoReader(path) as r:
        info = r.info()
        if info.get("GENERATOR") != GENERATOR:
            err(f"INFO/GENERATOR must be {GENERATOR!r}, got {info.get('GENERATOR')!r}")
        if int(info.get("FORMAT_VERSION", -1)) != FORMAT_VERSION:
            err(f"INFO/FORMAT_VERSION must be {FORMAT_VERSION}, got {info.get('FORMAT_VERSION')}")
        ndim = int(info.get("SPATIAL_DIM", 0))
        if ndim not in (1, 2, 3):
            err(f"INFO/SPATIAL_DIM must be 1|2|3, got {ndim}")

        # Partition manifest (parallel output). Only present/meaningful when this is
        # one ".part-N.ladruno" of a partitioned set; PARTITION_ID must index into
        # the 0-based-contiguous NUM_PARTITIONS the apeGmsh stitcher globs for.
        if int(info.get("PARTITIONED", 0)):
            pid = int(info.get("PARTITION_ID", -1))
            nparts = int(info.get("NUM_PARTITIONS", 0))
            if not (0 <= pid < nparts):
                err(f"INFO: PARTITION_ID {pid} not in [0, NUM_PARTITIONS={nparts})")

        stages = r.stages()
        if not stages:
            err("no MODEL_STAGE[*] group present")

        for stage in stages:
            sa = r.stage_attrs(stage)
            if sa.get("KIND") not in ("transient", "static", "eigen"):
                err(f"{stage}: KIND must be transient|static|eigen, got {sa.get('KIND')!r}")

            nid, ncoord = r.nodes(stage)
            if ncoord.shape[0] != nid.shape[0]:
                err(f"{stage}: NODES COORDINATES rows {ncoord.shape[0]} != ID {nid.shape[0]}")
            if ncoord.ndim != 2 or ncoord.shape[1] != ndim:
                err(f"{stage}: NODES COORDINATES must be [n x {ndim}], got {ncoord.shape}")

            # element groups: BASIS + QUADRATURE invariants
            groups = r.element_groups(stage)
            for name, g in groups.items():
                if g["layout"] == "csr":
                    # §3.4: OFFSETS [nElem+1] starts at 0, never decreases, ends at
                    # len(NODES); NUM_CTRL=-1; no basis-dependent data (QUADRATURE /
                    # GLOBAL_GP_COORDS / CTRL_WEIGHT) since the node count varies.
                    off, nel = g["offsets"], g["ids"].shape[0]
                    if off.shape[0] != nel + 1:
                        err(f"{stage}/{name}: CONNECTIVITY_OFFSETS len {off.shape[0]} != nElem+1 ({nel + 1})")
                    elif nel and (off[0] != 0 or np.any(np.diff(off) < 0)
                                  or off[-1] != g["nodes"].shape[0]):
                        err(f"{stage}/{name}: CONNECTIVITY_OFFSETS not 0..len(NODES) non-decreasing")
                    if g["num_ctrl"] != -1:
                        err(f"{stage}/{name}: CSR group needs NUM_CTRL=-1, got {g['num_ctrl']}")
                    if "global_gp_coords" in g or g["rational"]:
                        err(f"{stage}/{name}: CSR group carries GLOBAL_GP_COORDS/RATIONAL")
                    continue
                if g["connectivity"].shape[1] != 1 + g["num_ctrl"]:
                    err(f"{stage}/{name}: CONNECTIVITY cols {g['connectivity'].shape[1]} "
                        f"!= 1+NUM_CTRL ({1 + g['num_ctrl']})")
                if g["gp_param"] is None:
                    # QUADRATURE-tolerant (schema-v1 decision): an untabulated rule
                    # legitimately emits no QUADRATURE. Warn (stderr), do not fail.
                    print(f"WARN {stage}/{name}: no QUADRATURE group (untabulated "
                          f"rule); skipping GP checks", file=sys.stderr)
                else:
                    # NDIR is authoritative for the parametric dim (D4), NOT len(ORDER)
                    # -- simplices have ndir 2/3 while ORDER stays total-degree.
                    ndir = g["ndir"]
                    if g["gp_param"].shape != (g["num_gp"], ndir):
                        err(f"{stage}/{name}: GP_PARAM {g['gp_param'].shape} != "
                            f"(NUM_GP={g['num_gp']}, NDIR={ndir})")
                    if g["param_domain"] == "bary" and ndir not in (2, 3):
                        err(f"{stage}/{name}: bary PARAM_DOMAIN needs NDIR in (2,3), got {ndir}")
                    if g["gp_weight"] is not None and g["gp_weight"].shape[0] != g["num_gp"]:
                        err(f"{stage}/{name}: GP_WEIGHT len {g['gp_weight'].shape[0]} != NUM_GP")
                    # GLOBAL_GP_COORDS (D2) optional; if present, [nElem x NUM_GP*ndim].
                    if "global_gp_coords" in g:
                        nelem = g["connectivity"].shape[0]
                        want = (nelem, g["num_gp"] * ndim)
                        if g["global_gp_coords"].shape != want:
                            err(f"{stage}/{name}: GLOBAL_GP_COORDS {g['global_gp_coords'].shape}"
                                f" != (nElem, NUM_GP*ndim)={want}")
                if g["topology"] not in (
                    "line", "tri", "quad", "tet", "hex", "wedge", "pyramid", "custom"
                ):
                    err(f"{stage}/{name}: unknown TOPOLOGY {g['topology']!r}")
                if g["rational"]:
                    if "ctrl_weight" not in g:
                        err(f"{stage}/{name}: RATIONAL=1 but no CTRL_WEIGHT")
                    elif g["ctrl_weight"].shape != (
                        g["connectivity"].shape[0], g["num_ctrl"]
                    ):
                        err(f"{stage}/{name}: CTRL_WEIGHT {g['ctrl_weight'].shape} != "
                            f"(nElem, NUM_CTRL)")

            # local axes alignment
            for name, la in r.local_axes(stage).items():
                if la["frame"].shape != (la["id"].shape[0], 4):
                    err(f"{stage}/LOCAL_AXES/{name}: FRAME {la['frame'].shape} != (nE, 4)")

            # section assignments
            for name, s in r.section_assignments(stage).items():
                if s["kind"] not in ("fiber", "resultant"):
                    err(f"{stage}/SECTION_ASSIGNMENTS/{name}: KIND must be fiber|resultant")
                if s["kind"] == "fiber":
                    if "fiber_data" not in s:
                        err(f"{stage}/SECTION_ASSIGNMENTS/{name}: fiber kind needs FIBER_DATA")
                    elif s["fiber_data"].shape[0] != s["fiber_materials"].shape[0]:
                        err(f"{stage}/SECTION_ASSIGNMENTS/{name}: FIBER_DATA rows != "
                            f"FIBER_MATERIALS")

            _validate_element_results(r, stage, err)
            _validate_nodal_results(r, stage, err)
            _validate_envelopes(r, stage, err)
            _validate_partition_reduction(r, stage, err, bool(int(info.get("PARTITIONED", 0))))

        _validate_particles(r, info, ndim, err)

    return problems


PARTITION_REDUCTIONS = ("NONE", "SUM", "UNSUPPORTED")


def _validate_partition_reduction(r: LadrunoReader, stage: str, err, partitioned: bool) -> None:
    """PARTITION_REDUCTION (WP-126, schema §7.1): how a reader combines a result across
    partition files. Optional (files written before WP-126 lack it); when present it must
    be NONE|SUM|UNSUPPORTED. In a PARTITIONED file an envelope group must be NONE: the
    recorder refuses to envelope a partial, because a per-partition extreme cannot be
    recombined."""
    path = f"{stage}/RESULTS"
    if path not in r.f:
        return

    def visit(name, obj):
        if not hasattr(obj, "attrs") or "PARTITION_REDUCTION" not in obj.attrs:
            return
        val = _attr(obj, "PARTITION_REDUCTION")
        ctx = f"{path}/{name}"
        if val not in PARTITION_REDUCTIONS:
            err(f"{ctx}: PARTITION_REDUCTION {val!r} not in {PARTITION_REDUCTIONS}")
        elif partitioned and name.startswith("ENVELOPES/") and val != "NONE":
            err(f"{ctx}: envelope of a {val} result in a partitioned file "
                f"(a per-partition extreme of a partial is unrecoverable)")

    r.f[path].visititems(visit)


def _validate_envelopes(r: LadrunoReader, stage: str, err) -> None:
    """ENVELOPES/<family>/<name>/{ID,MIN,MAX,ABSMAX,ARG_STEP} (ADR D7): the four
    accumulators are [nIds x nComp], and componentwise MIN<=MAX with
    ABSMAX==max(|MIN|,|MAX|) (the abs-extreme over time equals the larger magnitude
    of the running min/max)."""
    path = f"{stage}/RESULTS/ENVELOPES"
    if path not in r.f:
        return
    for fam in r.f[path]:
        for name, g in _iter_envelope_leaves(r.f[path][fam]):
            ctx = f"{stage}/ENVELOPES/{fam}/{name}"
            missing = [d for d in ("ID", "MIN", "MAX", "ABSMAX", "ARG_STEP") if d not in g]
            if missing:
                err(f"{ctx}: missing {missing}")
                continue
            n = g["ID"].shape[0]
            mn, mx, am = g["MIN"][...], g["MAX"][...], g["ABSMAX"][...]
            for dname, arr in (("MIN", mn), ("MAX", mx), ("ABSMAX", am),
                               ("ARG_STEP", g["ARG_STEP"][...])):
                if arr.ndim != 2 or arr.shape[0] != n:
                    err(f"{ctx}/{dname}: shape {arr.shape} rows != nIds {n}")
            if mn.shape == mx.shape == am.shape:
                if np.any(mn > mx + 1e-12):
                    err(f"{ctx}: MIN > MAX somewhere")
                if not np.allclose(am, np.maximum(np.abs(mn), np.abs(mx)), atol=1e-9):
                    err(f"{ctx}: ABSMAX != max(|MIN|,|MAX|)")
            # self-describing component names (finding (h)): if present, COMPONENTS
            # count must equal the number of columns (nComp).
            if "COMPONENTS" in g.attrs and mn.ndim == 2:
                comps = [c for c in _attr(g, "COMPONENTS").split(",") if c]
                if comps and len(comps) != mn.shape[1]:
                    err(f"{ctx}: COMPONENTS count {len(comps)} != nComp {mn.shape[1]}")
            # Element envelopes carry the structured per-column COLUMN_MAP (the
            # authoritative gauss/section/fiber column map, since COMPONENTS is empty
            # for element results). Validate it the same way the time-series
            # ON_ELEMENTS path is validated, and that it spans every column.
            if "COLUMN_MAP" in g and mn.ndim == 2:
                _validate_column_map(g["COLUMN_MAP"], mn.shape[1], ctx, err)


# ---------------------------------------------------------------------------
# Round-trip ORACLE (D5) -- the write-time geometry self-check
# ---------------------------------------------------------------------------


def round_trip_oracle(path: str, tol: float = 1e-12) -> list[str]:
    """Mandatory write->read->reconstruct geometry check (schema-v1 D5).

    For every element group that carries BOTH a GP_PARAM table and the static
    ``GLOBAL_GP_COORDS`` (the belt-and-suspenders D2 dataset), independently
    reconstruct ``x(GP_PARAM[k]) = sum_i N_i(xi_k) X_i`` from the file's own
    MODEL/NODES + CONNECTIVITY using ``ladruno_basis.shape_functions`` and compare
    to ``GLOBAL_GP_COORDS[k]`` (<= ``tol``). This catches GP-ordering and basis
    bugs *at write time*: if the recorder's C++ ``computeGlobalGP`` and this
    independent Python basis disagree, one is wrong.

    Groups with no GLOBAL_GP_COORDS or no QUADRATURE are skipped (nothing to
    cross-check); a topology the oracle basis does not cover (e.g. a custom-rule
    bernstein/IGA element whose own gate validates it) is warned and skipped, not
    failed. Returns a list of problems; empty == oracle-clean.
    """
    problems: list[str] = []
    with LadrunoReader(path) as r:
        ndim = int(r.info().get("SPATIAL_DIM", 0))
        for stage in r.stages():
            nid, ncoord = r.nodes(stage)
            nid = np.asarray(nid).reshape(-1)
            coord_of = {int(t): np.asarray(ncoord[i], dtype=float) for i, t in enumerate(nid)}
            for name, g in r.element_groups(stage).items():
                if (g.get("global_gp_coords") is None or g["gp_param"] is None
                        or g["layout"] == "csr"):
                    continue
                topo = g["topology"]
                conn = g["connectivity"]
                num_gp = g["num_gp"]
                nelem = conn.shape[0]
                num_nodes = conn.shape[1] - 1
                ggp = np.asarray(g["global_gp_coords"], dtype=float)
                if ggp.size != nelem * num_gp * ndim:
                    problems.append(
                        f"{stage}/{name}: GLOBAL_GP_COORDS size {ggp.size} != "
                        f"nElem*NUM_GP*ndim ({nelem}*{num_gp}*{ndim})")
                    continue
                ggp3 = ggp.reshape(nelem, num_gp, ndim)
                max_err = 0.0
                try:
                    for e in range(nelem):
                        X = np.array([coord_of[int(t)][:ndim] for t in conn[e, 1:]])
                        for k in range(num_gp):
                            N = ladruno_basis.shape_functions(topo, num_nodes, g["gp_param"][k])
                            x_rec = N @ X
                            max_err = max(max_err, float(np.max(np.abs(x_rec - ggp3[e, k]))))
                except ValueError as exc:
                    print(f"WARN {stage}/{name}: oracle skipped ({exc})", file=sys.stderr)
                    continue
                if max_err > tol:
                    problems.append(
                        f"{stage}/{name}: round-trip x(GP_PARAM) vs GLOBAL_GP_COORDS "
                        f"max_err={max_err:.3e} > tol {tol:.0e}")
    return problems


def _validate_nodal_results(r: LadrunoReader, stage: str, err) -> None:
    path = f"{stage}/RESULTS/ON_NODES"
    if path not in r.f:
        return
    for rname in r.f[path]:
        grp = r.f[path][rname]
        ids = grp["ID"][...]
        ncomp = len([c for c in _attr(grp, "COMPONENTS").split(",") if c])
        _check_data_shape(grp, ids.shape[0], ncomp, err, f"{stage}/ON_NODES/{rname}")


def _validate_column_map(cm, num_cols: int, ctx: str, err) -> None:
    """Validate a COLUMN_MAP group (schema §7.2), shared by the time-series
    ON_ELEMENTS path and the element ENVELOPES path: every per-block array has the
    same length k, COMP_NAMES carries k lines, and sum(MULTIPLICITY*NUM_COMP) spans
    exactly num_cols (the result's flat column count)."""
    mult = cm["MULTIPLICITY"][...]
    ncomp = cm["NUM_COMP"][...]
    k = len(ncomp)
    for arr_name in ("LEVELS", "GAUSS_ID", "SECTION_TAG", "FIBER_ID", "MULTIPLICITY"):
        if len(cm[arr_name][...]) != k:
            err(f"{ctx}: COLUMN_MAP/{arr_name} len != {k}")
    if len(_decode_lines(_attr(cm, "COMP_NAMES"))) != k:
        err(f"{ctx}: COMP_NAMES lines != {k} blocks")
    expect = int(np.sum(mult * ncomp))
    if expect != num_cols:
        err(f"{ctx}: NUM_COLUMNS {num_cols} != sum(MULT*NUM_COMP) {expect}")


def _validate_element_results(r: LadrunoReader, stage: str, err) -> None:
    path = f"{stage}/RESULTS/ON_ELEMENTS"
    if path not in r.f:
        return
    for rname in r.f[path]:
        rgrp = r.f[path][rname]
        for bname in rgrp:
            bucket = rgrp[bname]
            if "COLUMN_MAP" not in bucket:
                continue
            num_cols = int(_attr(bucket, "NUM_COLUMNS"))
            ctx = f"{stage}/ON_ELEMENTS/{rname}/{bname}"
            _validate_column_map(bucket["COLUMN_MAP"], num_cols, ctx, err)
            ids = bucket["ID"][...]
            _check_data_shape(bucket, ids.shape[0], num_cols, err,
                              f"{stage}/ON_ELEMENTS/{rname}/{bname}")
            if "SECTION_MAP" in bucket:
                # SECTION_MAP is [nElem x NUM_GP]; resolve NUM_GP from the model group
                sm = bucket["SECTION_MAP"][...]
                if sm.shape[0] != ids.shape[0]:
                    err(f"{stage}/ON_ELEMENTS/{rname}/{bname}: SECTION_MAP rows "
                        f"{sm.shape[0]} != nElem {ids.shape[0]}")


# ---------------------------------------------------------------------------
# MPM particle group (WP-232a, contract 2) -- FORMAT_MINOR_VERSION 2
# ---------------------------------------------------------------------------
# The rules are Ladruno_implementation/232a_mpm_contracts.md §2; every message starts
# with the rule's code (C2-...), so a test can name the rule it expects to fire.

H5Z_ALLOWED_FILTERS = {1: "deflate", 2: "shuffle", 3: "fletcher32"}  # H5Z_FILTER_* ids

PARTICLE_INDEX = ("STEP", "TIME", "OFFSETS", "ID")
PARTICLE_FIELD_NAME = re.compile(r"^(?:[A-Z][A-Z0-9_]*|STATE_[A-Za-z0-9_]+)$")
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
MAX_SAFE_ID = 2**53 - 1

#: reserved field -> (COMPONENTS for ndm 2, COMPONENTS for ndm 3)  (232a §2.6)
PARTICLE_FIELDS = {
    "POSITION": ("x,y", "x,y,z"),
    "VELOCITY": ("Vx,Vy", "Vx,Vy,Vz"),
    "ACCELERATION": ("Ax,Ay", "Ax,Ay,Az"),
    "STRESS": ("sigma11,sigma22,sigma12",
               "sigma11,sigma22,sigma33,sigma12,sigma23,sigma13"),
    "STRESS33": ("sigma33", None),          # plane strain only; in 3D it is in STRESS
    "STRAIN": ("eps11,eps22,gamma12", "eps11,eps22,eps33,gamma12,gamma23,gamma13"),
    "VOLUME": ("V", "V"),
    "GIMP_HALF_LENGTH": ("lx,ly", "lx,ly,lz"),
    "DEFORMATION_GRADIENT": ("F11,F12,F21,F22", "F11,F12,F13,F21,F22,F23,F31,F32,F33"),
}
FIELD_ATTRS = ("DISPLAY_NAME", "COMPONENTS", "TYPE", "DATA_TYPE", "PARTITION_REDUCTION")
#: MODE -> the stage KIND it runs under (232a §2.3)
MODE_KIND = {"explicit": "transient", "implicit": "transient", "quasi-static": "static"}
#: the empty FE skeleton of a particle-only stage, as LadrunoRecorder writes it (WP-165 MP-8)
EMPTY_RESULT_GROUPS = ("ON_NODES", "ON_ELEMENTS", "ON_DOMAIN", "ON_REGIONS")


def int_attr_ok(obj, name: str, width: int = 4) -> bool:
    """An integer attribute of the stated width (int32 unless the contract says int64)."""
    v = np.asarray(obj.attrs[name])
    return v.dtype.kind in "iu" and v.dtype.itemsize == width and v.size >= 1


def _stage_id(name: str) -> int:
    """MODEL_STAGE[<id>] -> id (stages sort numerically, schema §2a)."""
    return int(name[name.index("[") + 1:name.index("]")])


def _dtype_is(ds, *want: str) -> bool:
    return any(ds.dtype == np.dtype(w) for w in want)


def filters_of(ds) -> list[int]:
    """HDF5 filter ids on a dataset's creation plist (empty for contiguous)."""
    dcpl = ds.id.get_create_plist()
    return [dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())]


def check_filters(ds, ctx: str, err, code: str) -> None:
    """Only shuffle / deflate / fletcher32: what h5wasm and the fork's HDF5 both read."""
    bad = [f for f in filters_of(ds) if f not in H5Z_ALLOWED_FILTERS]
    if bad:
        err(f"{code}: {ctx}: filters {bad} outside shuffle/deflate/fletcher32")


def str_dataset(ds) -> list[str]:
    """A string dataset, fixed- or variable-length, as a list of str."""
    return [str(x) for x in ds.asstr()[...].reshape(-1)]


def _validate_particles(r: LadrunoReader, info: dict, ndim: int, err) -> None:
    for stage in r.stages():
        has_model = f"{stage}/MODEL/PARTICLES" in r.f
        has_res = f"{stage}/RESULTS/ON_PARTICLES" in r.f
        if has_model != has_res:
            err(f"C2-PAIR: {stage}: MODEL/PARTICLES and RESULTS/ON_PARTICLES must come together")
    stages = [s for s in r.particle_stages() if f"{s}/RESULTS/ON_PARTICLES" in r.f]
    if not stages:
        return
    minor = int(info.get("FORMAT_MINOR_VERSION", 0))
    if minor < FORMAT_MINOR_VERSION_PARTICLES:
        err(f"C2-MINOR: INFO/FORMAT_MINOR_VERSION is {minor}; a file with a particle "
            f"group needs >= {FORMAT_MINOR_VERSION_PARTICLES}")
    if int(info.get("PARTITIONED", 0)):
        err("C2-PART: a partitioned part file carries a particle group (serial only)")
    if ndim not in (2, 3):
        err(f"C2-NDM: INFO/SPATIAL_DIM {ndim} must be 2 or 3 for particles")
        return
    removed: set[int] = set()      # ids that left the epoch: never seen again (C2-REUSE)
    alive: set[int] = set()
    epoch = None
    for stage in stages:
        sg = r.f[stage]
        if "PARTICLE_EPOCH" not in sg.attrs or not int_attr_ok(sg, "PARTICLE_EPOCH"):
            err(f"C2-EPOCH: {stage}: PARTICLE_EPOCH (int32) missing")
            ep = epoch if epoch is not None else 0
        else:
            ep = int(_attr(sg, "PARTICLE_EPOCH"))
            if ep < 0 or (epoch is not None and ep < epoch):
                err(f"C2-EPOCH: {stage}: PARTICLE_EPOCH {ep} is negative or decreases")
        if ep != epoch:            # `mpm reset` starts a new epoch: the id rules restart
            alive, removed = set(), set()
        epoch = ep
        alive, removed = _validate_particle_stage(r, stage, ndim, err, alive, removed)


def _validate_particle_model(r, stage, ndim, err):
    """MODEL/PARTICLES: attributes, GRID, BODIES, registry. Returns (grid lo, hi, ids)."""
    mp = r.f[f"{stage}/MODEL/PARTICLES"]
    ctx = f"{stage}/MODEL/PARTICLES"
    a = {k: _attr(mp, k) for k in mp.attrs}

    sid = a.get("SOURCE_SET_ID")     # optional: absent when no contract-1 file was loaded
    if sid is not None and (not isinstance(sid, str) or not UUID4.match(sid)):
        err(f"C2-SRC: {ctx}: SOURCE_SET_ID {sid!r} is not a canonical uuid4")
    if "SOURCE_SET_IDS" in a:
        try:
            ok = isinstance(json.loads(a["SOURCE_SET_IDS"]), list)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            err(f"C2-SRC: {ctx}: SOURCE_SET_IDS is not a JSON array")
    if "PATTERN_TAG" not in a or not int_attr_ok(mp, "PATTERN_TAG"):
        err(f"C2-PATTERN: {ctx}: PATTERN_TAG (int32) missing")
    if a.get("MODE") not in MODE_KIND:
        err(f"C2-MODE: {ctx}: MODE {a.get('MODE')!r} not explicit|implicit|quasi-static")
    else:
        kind = r.stage_attrs(stage).get("KIND")
        if kind in ("transient", "static") and MODE_KIND[a["MODE"]] != kind:
            err(f"C2-MODEKIND: {ctx}: MODE {a['MODE']} runs under KIND "
                f"{MODE_KIND[a['MODE']]}, the stage says {kind}")
    thk = a.get("PLANE_STRAIN_THICKNESS")
    if ndim == 2 and (thk is None or not np.isfinite(thk) or thk <= 0):
        err(f"C2-THK: {ctx}: PLANE_STRAIN_THICKNESS must be > 0 when SPATIAL_DIM = 2")
    if ndim == 3 and thk is not None:
        err(f"C2-THK: {ctx}: PLANE_STRAIN_THICKNESS present in 3D")
    for u in ("UNIT_LENGTH", "UNIT_FORCE", "UNIT_TIME"):
        if not isinstance(a.get(u), str) or not a[u]:
            err(f"C2-UNIT: {ctx}: {u} missing or empty")

    lo = hi = None
    if "GRID" not in mp:
        err(f"C2-GRID: {ctx}: GRID group missing")
    else:
        lo, hi = check_grid(mp["GRID"], ndim, f"{ctx}/GRID", err, "C2-GRID")

    body_ids: set[int] = set()
    if "BODIES" not in mp or "ID" not in mp["BODIES"] or "NAME" not in mp["BODIES"]:
        err(f"C2-BODY: {ctx}: BODIES/ID and BODIES/NAME are required")
    else:
        body_ids = check_bodies(mp["BODIES"], f"{ctx}/BODIES", err, "C2-BODY")

    reg_ids = np.zeros(0, dtype=np.int64)
    want = {"ID": "<i8", "BODY_ID": "<i4", "MATERIAL_TAG": "<i4", "MASS": "<f8"}
    missing = [k for k in want if k not in mp]
    if missing:
        err(f"C2-REG: {ctx}: registry datasets missing {missing}")
        return lo, hi, reg_ids
    for k, dt in want.items():
        if not _dtype_is(mp[k], dt):
            err(f"C2-REG: {ctx}/{k} dtype {mp[k].dtype} != {np.dtype(dt)}")
    shapes = {mp[k].shape for k in want}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        err(f"C2-REG: {ctx}: registry datasets must be 1-D and of one length")
        return lo, hi, reg_ids
    reg_ids = np.asarray(mp["ID"][...])
    if np.any(np.diff(reg_ids) <= 0):
        err(f"C2-REG: {ctx}/ID not strictly increasing")
    if reg_ids.size and (reg_ids.min() < 0 or reg_ids.max() > MAX_SAFE_ID):
        err(f"C2-REG: {ctx}/ID outside [0, 2^53-1]")
    if body_ids and not set(np.asarray(mp["BODY_ID"][...]).tolist()) <= body_ids:
        err(f"C2-REG: {ctx}/BODY_ID names a body not in BODIES/ID")
    if np.any(np.asarray(mp["MATERIAL_TAG"][...]) <= 0):
        err(f"C2-REG: {ctx}/MATERIAL_TAG must be > 0")
    m = np.asarray(mp["MASS"][...])
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        err(f"C2-REG: {ctx}/MASS must be finite and > 0")
    return lo, hi, reg_ids


def check_grid(g, ndim: int, ctx: str, err, code: str, basis_code: str | None = None):
    """GRID attributes (contract 1 §1.2 == contract 2 §2.4). Returns the box (lo, hi)."""
    lo = hi = None
    try:
        org = np.atleast_1d(np.asarray(_attr(g, "ORIGIN"), dtype=float))
        h = np.atleast_1d(np.asarray(_attr(g, "CELL_SIZE"), dtype=float))
        nc = np.atleast_1d(np.asarray(_attr(g, "NUM_CELLS")))
    except KeyError as exc:
        err(f"{code}: {ctx}: missing attribute {exc}")
        org = h = nc = None
    if org is not None:
        if not (org.shape == h.shape == nc.shape == (ndim,)):
            err(f"{code}: {ctx}: ORIGIN/CELL_SIZE/NUM_CELLS must each be [{ndim}]")
        elif nc.dtype.kind not in "iu":
            err(f"{code}: {ctx}: NUM_CELLS must be integer")
        elif not np.all(np.isfinite(org)) or not np.all(np.isfinite(h)) or np.any(h <= 0) \
                or np.any(nc < 1):
            err(f"{code}: {ctx}: non-finite origin, CELL_SIZE <= 0 or NUM_CELLS < 1")
        else:
            lo, hi = org, org + nc * h
    if "BASIS" not in g.attrs or _attr(g, "BASIS") not in ("gimp", "linear"):
        err(f"{basis_code or code}: {ctx}: BASIS must be gimp|linear")
    return lo, hi


def check_bodies(bg, ctx: str, err, code: str) -> set[int]:
    """BODIES/{ID int32, NAME str}: one length >= 1, unique ids, unique non-empty names."""
    bid = bg["ID"]
    names = str_dataset(bg["NAME"])
    if not _dtype_is(bid, "<i4"):
        err(f"{code}: {ctx}/ID dtype {bid.dtype} != int32")
    vals = np.asarray(bid[...]).reshape(-1)
    if len(vals) != len(names) or len(vals) < 1:
        err(f"{code}: {ctx}: ID and NAME lengths differ or are empty")
    if (len(set(vals.tolist())) != len(vals) or len(set(names)) != len(names)
            or "" in names or np.any(vals < 0)):
        err(f"{code}: {ctx}: ids (>= 0) and names must be unique, names non-empty")
    return set(vals.tolist())


def _validate_particle_stage(r, stage, ndim, err, alive, removed):
    kind = r.stage_attrs(stage).get("KIND")
    if kind not in ("transient", "static"):
        err(f"C2-KIND: {stage}: KIND {kind!r} must be transient|static for particles")
    lo, hi, reg_ids = _validate_particle_model(r, stage, ndim, err)
    _validate_particle_only_stage(r, stage, ndim, err)

    op = r.f[f"{stage}/RESULTS/ON_PARTICLES"]
    octx = f"{stage}/RESULTS/ON_PARTICLES"
    if "PARTITION_REDUCTION" not in op.attrs or _attr(op, "PARTITION_REDUCTION") != "NONE":
        err(f"C2-RED: {octx}: PARTITION_REDUCTION must be NONE")
    miss = [k for k in PARTICLE_INDEX if k not in op or not isinstance(op[k], h5py.Dataset)]
    if miss:
        err(f"C2-AXES: {octx}: index datasets missing {miss}")
        return alive, removed
    for k, dt, code in (("STEP", "<i4", "C2-AXES"), ("TIME", "<f8", "C2-AXES"),
                        ("OFFSETS", "<i8", "C2-OFF"), ("ID", "<i8", "C2-ID")):
        ds = op[k]
        if not _dtype_is(ds, dt):
            err(f"{code}: {octx}/{k} dtype {ds.dtype} != {np.dtype(dt)}")
        if ds.ndim != 1:
            err(f"{code}: {octx}/{k} must be 1-D")
        check_filters(ds, f"{octx}/{k}", err, "C2-FILTER")
    steps = np.asarray(op["STEP"][...]).reshape(-1)
    times = np.asarray(op["TIME"][...]).reshape(-1)
    offs = np.asarray(op["OFFSETS"][...]).reshape(-1)
    ids = np.asarray(op["ID"][...]).reshape(-1)
    T = steps.shape[0]
    if T < 1 or times.shape[0] != T:
        err(f"C2-AXES: {octx}: need T >= 1 outputs, STEP and TIME of one length")
        return alive, removed
    if np.any(np.diff(steps) <= 0):
        err(f"C2-AXES: {octx}/STEP not strictly increasing")
    # TIME may restart inside a stage (`loadConst -time 0`, `setTime`), as on ON_NODES;
    # readers key outputs by index / STEP, never by TIME.
    if np.any(~np.isfinite(times)):
        err(f"C2-AXES: {octx}/TIME not finite")
    if (offs.shape[0] != T + 1 or offs[0] != 0 or np.any(np.diff(offs) < 0)
            or offs[-1] != ids.shape[0]):
        err(f"C2-OFF: {octx}/OFFSETS must be [T+1], start at 0, never decrease, end at len(ID)")
        return alive, removed
    S = ids.shape[0]
    reg_set = set(reg_ids.tolist())
    for k in range(T):
        out = ids[offs[k]:offs[k + 1]]
        if np.any(np.diff(out) <= 0):
            err(f"C2-ID: {octx}: output {k} ids not strictly increasing (repeated or unsorted)")
        cur = set(out.tolist())
        if not cur <= reg_set:
            err(f"C2-ID: {octx}: output {k} has ids not in the registry {sorted(cur - reg_set)[:5]}")
        back = cur & removed
        if back:
            err(f"C2-REUSE: {octx}: output {k} reuses removed ids {sorted(back)[:5]}")
        removed |= alive - cur
        alive = cur

    if "POSITION" not in op:
        err(f"C2-POS: {octx}: POSITION is required")
    for name in op:
        if name in PARTICLE_INDEX:
            continue
        _validate_particle_field(op[name], name, f"{octx}/{name}", ndim, S, lo, hi, err)
    return alive, removed


def _validate_particle_only_stage(r, stage, ndim, err) -> None:
    """C2-EMPTY (232a §2.3): a particle stage with no FE nodes is written exactly as
    LadrunoRecorder writes an empty stage (LadrunoRecorder.cpp, writeModel + the WP-165
    MP-8 branch of writeModelNodes): EMPTY_PARTITION = 1, NODES/ID int32 [0 x 1],
    NODES/COORDINATES float64 [0 x ndm], an empty MODEL/ELEMENTS, and empty
    RESULTS/ON_NODES, ON_ELEMENTS, ON_DOMAIN, ON_REGIONS. Today's readers skip it as an
    empty stage. A stage with nodes must not claim EMPTY_PARTITION."""
    sg = r.f[stage]
    nodes = r.f.get(f"{stage}/MODEL/NODES")
    if nodes is None or "ID" not in nodes or "COORDINATES" not in nodes:
        err(f"C2-EMPTY: {stage}: MODEL/NODES/ID and COORDINATES are required")
        return
    n = nodes["ID"].shape[0] if nodes["ID"].ndim else 0
    empty = "EMPTY_PARTITION" in sg.attrs and int(_attr(sg, "EMPTY_PARTITION")) == 1
    if n > 0:
        if empty:
            err(f"C2-EMPTY: {stage}: EMPTY_PARTITION = 1 on a stage with {n} nodes")
        return
    probs = []
    if not empty:
        probs.append("EMPTY_PARTITION = 1 missing")
    if nodes["ID"].shape != (0, 1) or nodes["ID"].dtype != np.dtype("<i4"):
        probs.append(f"NODES/ID {nodes['ID'].dtype} {nodes['ID'].shape} != int32 (0, 1)")
    if nodes["COORDINATES"].shape != (0, ndim):
        probs.append(f"NODES/COORDINATES {nodes['COORDINATES'].shape} != (0, {ndim})")
    el = r.f.get(f"{stage}/MODEL/ELEMENTS")
    if el is None or len(el):
        probs.append("MODEL/ELEMENTS missing or not empty")
    for g in EMPTY_RESULT_GROUPS:
        grp = r.f.get(f"{stage}/RESULTS/{g}")
        if grp is None or len(grp):
            probs.append(f"RESULTS/{g} missing or not empty")
    if probs:
        err(f"C2-EMPTY: {stage}: particle-only stage is not the recorder's empty stage: "
            + "; ".join(probs))


def _validate_particle_field(item, name, fctx, ndim, S, lo, hi, err) -> None:
    if not isinstance(item, h5py.Group):
        err(f"C2-UNKNOWN: {fctx}: unexpected dataset in ON_PARTICLES")
        return
    if not PARTICLE_FIELD_NAME.match(name):
        err(f"C2-NAME: {fctx}: field name breaks the naming rule")
    fa = {k: _attr(item, k) for k in item.attrs}
    lacking = [k for k in FIELD_ATTRS if k not in fa]
    if lacking:
        err(f"C2-ATTR: {fctx}: attributes missing {lacking}")
    if fa.get("PARTITION_REDUCTION", "NONE") != "NONE":
        err(f"C2-RED: {fctx}: PARTITION_REDUCTION must be NONE")
    raw = str(fa.get("COMPONENTS", ""))
    comps = raw.split(",") if raw else []
    if not comps or "" in comps or len(set(comps)) != len(comps):
        err(f"C2-COMP: {fctx}: COMPONENTS {raw!r} has no, an empty or a repeated label")
    if name in PARTICLE_FIELDS:
        expect = PARTICLE_FIELDS[name][ndim - 2]
        if expect is None:
            err(f"C2-CAT: {fctx}: {name} is a plane-strain field; in 3D it is part of STRESS")
        elif ",".join(comps) != expect:
            err(f"C2-CAT: {fctx}: COMPONENTS {','.join(comps)!r} != {expect!r}")
    if "DATA" not in item or not isinstance(item["DATA"], h5py.Dataset):
        err(f"C2-DATA: {fctx}: DATA dataset missing")
        return
    data = item["DATA"]
    check_filters(data, f"{fctx}/DATA", err, "C2-FILTER")
    if name == "POSITION":
        if not _dtype_is(data, "<f8"):
            err(f"C2-DTYPE: {fctx}/DATA dtype {data.dtype}: POSITION is float64 only")
    elif not _dtype_is(data, "<f8", "<f4"):
        err(f"C2-DTYPE: {fctx}/DATA dtype {data.dtype} not float64|float32")
    if data.ndim != 2 or data.shape[0] != S:
        err(f"C2-DATA: {fctx}/DATA shape {data.shape} is not [S={S}, nComp]")
        return
    if data.shape[1] != len(comps):
        err(f"C2-COMP: {fctx}: {len(comps)} COMPONENTS for {data.shape[1]} DATA columns")
        return
    vals = np.asarray(data[...])
    # NaN is allowed only in STATE_* (the material does not answer the token)
    bad = np.isinf(vals) if name.startswith("STATE_") else ~np.isfinite(vals)
    if np.any(bad):
        err(f"C2-FINITE: {fctx}/DATA has non-finite values")
    if name == "POSITION" and lo is not None and vals.shape[1] == ndim:
        tol = 1e-9 * float(np.max(hi - lo))
        if np.any(vals < lo - tol) or np.any(vals > hi + tol):
            err(f"C2-POS: {fctx}: a position lies outside the grid box")
