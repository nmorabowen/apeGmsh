"""The MPCO -> ``.ladruno`` information-equality comparator (ADR 0122 D8).

Lifted unchanged from ``test_mpco_to_ladruno_oracle_live.py`` (#2120) so
the multi-stage oracle (#2177) runs the same comparator. :func:`compare`
reads both files through the fork's own ``ladruno_format`` (fetched at
``FORK_PIN`` by :mod:`._fork_spec`) and returns every mismatch it finds.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import h5py
import numpy as np

__all__ = ["INFO_EXCLUDED", "attr", "compare", "stages"]

_STAGE = "MODEL_STAGE["
# Values from one run are the same doubles in both files; the tolerance is
# the fork's own element parity check's (element_parity_check.py).
_ATOL, _RTOL = 1e-12, 1e-9
# The attributes both recorders write on a result (MPCO's own set; the fork
# may add PARTITION_REDUCTION, which an .mpco never carries).
_NODE_ATTRS = ("COMPONENTS", "DATA_TYPE", "DESCRIPTION", "DIMENSION",
               "DISPLAY_NAME", "TYPE")
_BUCKET_ATTRS = ("DATA_TYPE", "DISPLAY_NAME", "NUM_COLUMNS", "TYPE")


# ---------------------------------------------------------------------------
# The information-equality comparator
# ---------------------------------------------------------------------------


def _stages(f: "h5py.File") -> list[str]:
    return sorted((k for k in f if k.startswith(_STAGE)),
                  key=lambda n: int(n[len(_STAGE):-1]))


def _modal(name: str) -> bool:
    return name.startswith("MODES_OF_VIBRATION")


class _OnNodesView:
    """``reader.f`` with the modal groups hidden from one ``ON_NODES``.

    The fork's ``normalize_nodal`` iterates every ``ON_NODES`` group as a
    step series; a modal group (``DATA/STEP_k/MODE_i``) is compared on its
    own instead.
    """

    def __init__(self, f: "h5py.File", path: str) -> None:
        self._f, self._path = f, path

    def __contains__(self, path: str) -> bool:
        return path in self._f

    def __getitem__(self, path: str) -> Any:
        grp = self._f[path]
        if path != self._path:
            return grp
        return _Filtered(grp, [n for n in grp if not _modal(n)])


class _Filtered:
    def __init__(self, grp: "h5py.Group", names: "list[str]") -> None:
        self._grp, self._names = grp, names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __getitem__(self, name: str) -> Any:
        return self._grp[name]


def _close(a: "dict[Any, float]", b: "dict[Any, float]", what: str) -> list[str]:
    out = []
    if a.keys() != b.keys():
        only_a = sorted(set(a) - set(b), key=str)[:3]
        only_b = sorted(set(b) - set(a), key=str)[:3]
        out.append(f"{what}: keys differ (fork only {only_a}, converted "
                   f"only {only_b})")
    for k in a.keys() & b.keys():
        if not abs(a[k] - b[k]) <= _ATOL + _RTOL * abs(a[k]):
            out.append(f"{what}: {k} fork={a[k]!r} converted={b[k]!r}")
            break
    return out


def _arrays(a: Any, b: Any, what: str) -> list[str]:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or not np.array_equal(x, y):
        return [f"{what}: {x.shape} vs {y.shape} or values differ"]
    return []


def _attr(obj: Any, name: str) -> Any:
    if name not in obj.attrs:
        return None
    v = np.asarray(obj.attrs[name]).ravel()
    v = v[0] if v.size == 1 else v
    return v.decode() if isinstance(v, bytes) else v


#: INFO attributes left out of the comparison, each for a stated reason.
INFO_EXCLUDED = {
    # The converter writes minor 0 on purpose. Its KIND is per stage, but
    # minor 1 also promises the schema section 2a stage boundaries, and a
    # converted file keeps MPCO's stages, which split on a domain change
    # only.
    "FORMAT_MINOR_VERSION": "minor 0 by design",
    # An .mpco carries no run identity.
    "RUN_ID": "absent from MPCO", "RUN_ID_SCOPE": "absent from MPCO",
    # The two recorders of one binary stamp different constants (MPCO the
    # OPS_VERSION it was built with, Ladruno its own); the converter can
    # only copy MPCO's.
    "SOLVER_VERSION": "recorder-specific constant",
}


def _attrs(obj: Any) -> "dict[str, Any]":
    return {k: _attr(obj, k) for k in obj.attrs}


def _same(x: Any, y: Any) -> bool:
    return np.array_equal(np.asarray(x), np.asarray(y))


def compare(
    fork: Path, converted: Path, lf: ModuleType, program_kinds: "list[str]",
    *, kind_checked_elsewhere: "frozenset[str]" = frozenset(),
) -> "tuple[list[str], set[str]]":
    """``(mismatches, lossy)`` between a fork file and a conversion.

    ``lossy`` names, in ``PROVENANCE@unsupported`` vocabulary, what the
    fork file holds and the converted file lacks (``local_axes``,
    ``quadrature:<group>``). Every stage's converted ``KIND`` must equal
    the program's kind, and the fork's ``KIND`` must equal the converted
    one, except on the fork stages named in ``kind_checked_elsewhere``,
    whose fork-side equality a test of its own asserts (see
    :func:`test_modal_stage_kind_equals_the_fork_kind`). INFO attributes
    compare except :data:`INFO_EXCLUDED`; everything else must be equal.
    """
    problems: list[str] = []
    lossy: set[str] = set()
    with lf.LadrunoReader(str(fork)) as ra, lf.LadrunoReader(str(converted)) as rb:
        fa, fb = ra.f, rb.f
        ia, ib = _attrs(fa["INFO"]), _attrs(fb["INFO"])
        for key in sorted((set(ia) | set(ib)) - set(INFO_EXCLUDED)):
            if not _same(ia.get(key), ib.get(key)):
                problems.append(f"INFO @{key}: fork {ia.get(key)!r}, "
                                f"converted {ib.get(key)!r}")
        sa, sb = _stages(fa), _stages(fb)
        if len(sa) != len(sb):
            return ([f"stage count: fork {len(sa)}, converted {len(sb)}"],
                    lossy)
        if len(program_kinds) != len(sb):
            problems.append(f"{len(program_kinds)} program kinds for "
                            f"{len(sb)} stages")
        for k, (ga, gb) in enumerate(zip(sa, sb)):
            where = f"{ga}|{gb}"
            aa, ab = _attrs(fa[ga]), _attrs(fb[gb])
            for key in sorted((set(aa) | set(ab)) - {"KIND"}):
                if not _same(aa.get(key), ab.get(key)):
                    problems.append(f"{where} @{key}: fork {aa.get(key)!r}, "
                                    f"converted {ab.get(key)!r}")
            ka, kb = aa.get("KIND"), ab.get("KIND")
            want = program_kinds[k] if k < len(program_kinds) else None
            if kb != want:
                problems.append(f"{where} @KIND: converted {kb!r}, program "
                                f"{want!r}")
            if ga not in kind_checked_elsewhere and ka != kb:
                problems.append(f"{where} @KIND: fork {ka!r}, converted "
                                f"{kb!r}")
            ida, xa = ra.nodes(ga)
            idb, xb = rb.nodes(gb)
            problems += _arrays(ida, idb, f"{where} NODES/ID")
            problems += _arrays(xa, xb, f"{where} NODES/COORDINATES")
            ea, eb = ra.element_groups(ga), rb.element_groups(gb)
            if list(ea) != list(eb):
                problems.append(f"{where} ELEMENTS: {list(ea)} vs {list(eb)}")
            for name in ea.keys() & eb.keys():
                problems += _arrays(ea[name]["connectivity"],
                                    eb[name]["connectivity"],
                                    f"{where} {name} CONNECTIVITY")
                if eb[name]["gp_param"] is None:
                    if ea[name]["gp_param"] is not None:
                        lossy.add(f"quadrature:{name}")
                    continue
                for key in ("topology", "family", "param_domain", "num_gp",
                            "ndir", "num_ctrl", "rational"):
                    if ea[name][key] != eb[name][key]:
                        problems.append(
                            f"{where} {name} {key}: {ea[name][key]!r} vs "
                            f"{eb[name][key]!r}")
                problems += _arrays(ea[name]["gp_param"], eb[name]["gp_param"],
                                    f"{where} {name} GP_PARAM")
            if ra.local_axes(ga) and not rb.local_axes(gb):
                lossy.add("local_axes")
            sec_a, sec_b = ra.section_assignments(ga), rb.section_assignments(gb)
            if list(sec_a) != list(sec_b):
                problems.append(f"{where} SECTIONS: {list(sec_a)} vs {list(sec_b)}")
            for name in sec_a.keys() & sec_b.keys():
                for key in ("kind", "assignment", "fiber_data", "fiber_materials"):
                    problems += _arrays(sec_a[name].get(key), sec_b[name].get(key),
                                        f"{where} {name} {key}")
            problems += _results(fa, fb, ga, gb, lf, where)
    return problems, lossy


def _results(
    fa: "h5py.File", fb: "h5py.File", ga: str, gb: str, lf: ModuleType,
    where: str,
) -> list[str]:
    problems: list[str] = []
    na, nb = f"{ga}/RESULTS/ON_NODES", f"{gb}/RESULTS/ON_NODES"
    names_a = list(fa[na]) if na in fa else []
    names_b = list(fb[nb]) if nb in fb else []
    if names_a != names_b:
        problems.append(f"{where} ON_NODES: {names_a} vs {names_b}")
    ea, eb = f"{ga}/RESULTS/ON_ELEMENTS", f"{gb}/RESULTS/ON_ELEMENTS"
    buckets_a = [(t, b) for t in (fa[ea] if ea in fa else ()) for b in fa[ea][t]]
    buckets_b = [(t, b) for t in (fb[eb] if eb in fb else ()) for b in fb[eb][t]]
    if buckets_a != buckets_b:
        problems.append(f"{where} buckets: {sorted(set(buckets_a) ^ set(buckets_b))}")
    # Step series: the fork's normalizers, keyed by information.
    for normalize, label in ((lf.normalize_nodal, "nodal"),
                             (lf.normalize_element, "element")):
        va = normalize(SimpleNamespace(f=_OnNodesView(fa, na)), ga)
        vb = normalize(SimpleNamespace(f=_OnNodesView(fb, nb)), gb)
        if not va:
            problems.append(f"{where} {label}: the fork file normalizes to "
                            "nothing; the oracle would compare no data")
        problems += _close(va, vb, f"{where} {label}")
    # Self-description: result attributes, and each bucket's COLUMN_MAP.
    for name in (n for n in names_a if n in names_b):
        for key in _NODE_ATTRS:
            if _attr(fa[f"{na}/{name}"], key) != _attr(fb[f"{nb}/{name}"], key):
                problems.append(f"{where} ON_NODES/{name} @{key}")
    for token, bucket in (k for k in buckets_a if k in set(buckets_b)):
        pa, pb = f"{ea}/{token}/{bucket}", f"{eb}/{token}/{bucket}"
        for key in _BUCKET_ATTRS:
            if _attr(fa[pa], key) != _attr(fb[pb], key):
                problems.append(f"{where} {token}/{bucket} @{key}")
        ca, cb = fa[f"{pa}/COLUMN_MAP"], fb[f"{pb}/COLUMN_MAP"]
        if sorted(ca) != sorted(cb):
            problems.append(f"{where} {token}/{bucket} COLUMN_MAP children")
            continue
        for key in ca:
            problems += _arrays(ca[key], cb[key],
                                f"{where} {token}/{bucket} COLUMN_MAP/{key}")
        if _attr(ca, "COMP_NAMES") != _attr(cb, "COMP_NAMES"):
            problems.append(f"{where} {token}/{bucket} @COMP_NAMES")
    # STEP / TIME axes of every step series.
    for base_a, base_b, items in (
            (na, nb, [(n,) for n in names_a if not _modal(n)]),
            (ea, eb, buckets_a)):
        for item in items:
            pa, pb = "/".join((base_a, *item)), "/".join((base_b, *item))
            if pb not in fb:
                continue
            for axis in ("STEP", "TIME"):
                problems += _arrays(fa[pa][axis], fb[pb][axis],
                                    f"{where} {'/'.join(item)} {axis}")
    # Modes: every MODE_i dataset and its eigen attributes.
    for name in (n for n in names_a if _modal(n) and n in names_b):
        da, db = fa[f"{na}/{name}/DATA"], fb[f"{nb}/{name}/DATA"]
        problems += _arrays(fa[f"{na}/{name}/ID"], fb[f"{nb}/{name}/ID"],
                            f"{where} {name} ID")
        if sorted(da) != sorted(db):
            problems.append(f"{where} {name}: steps {sorted(da)} vs {sorted(db)}")
            continue
        for step in da:
            if sorted(da[step]) != sorted(db[step]):
                problems.append(f"{where} {name}/{step}: modes differ")
                continue
            for mode in da[step]:
                problems += _arrays(da[step][mode], db[step][mode],
                                    f"{where} {name}/{step}/{mode}")
                for key in ("LAMBDA", "OMEGA", "FREQUENCY", "PERIOD", "MODE"):
                    if _attr(da[step][mode], key) != _attr(db[step][mode], key):
                        problems.append(f"{where} {name}/{step}/{mode} @{key}")
    return problems


#: Public names for the oracle files.
stages = _stages
attr = _attr
