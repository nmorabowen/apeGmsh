"""The ``/spectra`` cases: one spectrum, one direction, per-mode increments (ADR 0130 D5).

A case is what ``apeSees.spectrum_after_gravity`` computed for one
:class:`~apeGmsh.opensees.analysis.spectrum.DesignSpectrum`: its modes (``λ``,
the period ``Sa`` was read at, ``Γ``, ``Sa``, the damping a combination uses),
each extractor's per-mode increment about the reference state (``RESULTS``,
``S = M``, row ``i`` is mode ``modes/mode[i]``) and the reference state's own
values (``reference/RESULTS``, ``S = 1``). The ADR 0124 driver refuses
recorders, so nothing here derives from a ``.ladruno``; the zone goes stale
with the model hashes only (A1, A6).

**What a case holds** (A7). A labelled extractor is written as a domain
quantity, ``ON_DOMAIN/<extractor name>``, with its labels as components. So are
an unlabelled one that returns a single value (component ``1``) and the
driver's ``base_shear`` (components ``1..ndm``, the directions). An unlabelled
extractor with more values (an
:class:`~apeGmsh.opensees.analysis.spectrum.ElementResponse`) is refused unless
named in ``omit=``: written flat, it would lose which element and which force
each value is. An extractor whose labels are a
:class:`~apeGmsh.opensees.analysis.spectrum.NodeDisp`'s is a nodal field: it is
written as ``ON_NODES/DISPLACEMENT`` (the recorder spelling, ``Ux,Uy,Uz``) when
named in ``nodal=``, left out when named in ``omit=``, and refused otherwise, so
a full field is never written by default and never dropped silently. Only
translational displacement is written as a nodal field here. A 1.0.0 case holds
``ON_NODES`` and ``ON_DOMAIN`` only; the reader refuses any other family.

**Units of the spectrum.** ``spectrum/Sa`` is the spectrum as given, without
``@scale``; the per-mode ``sa`` column is the ``Sa`` the driver applied,
``@scale`` included. A table spectrum stores its table; a function spectrum is
sampled at the periods its modes read ``Sa`` at.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import h5py
import numpy as np

from ...opensees._internal.schema_version import SPECTRA_SCHEMA_VERSION
from ...opensees.analysis.spectrum import BASE_SHEAR, SpectrumModalResult
from ._descriptor import Refusal, decode_cell
from ._errors import ZoneRefused
from ._fields import (
    EntryRefused,
    _f8,
    _link_problem,
    _stack_axes,
    _token_problem,
    _write_id,
)
from ._modes import KINDS
from ._names import name_problem, text_or_empty

__all__ = ["CaseQuantity", "SpectraIndex", "SpectrumCase", "spectrum_cases"]

_STR = h5py.string_dtype(encoding="utf-8")
_CASE = re.compile(r"s(\d+)\Z")
_ON = {"domain": "ON_DOMAIN", "nodes": "ON_NODES"}
_TARGET = {v: k for k, v in _ON.items()}
#: ``NodeDisp`` labels (``spectrum.NodeDisp.labels``): ``node<n>.u<d>``.
_DISP_LABEL = re.compile(r"node(\d+)\.u(\d+)\Z")
#: ``NodeReaction`` labels: ``node<n>.R<d>``.
_REACTION_LABEL = re.compile(r"node(\d+)\.R(\d+)\Z")
_DISPLACEMENT = "DISPLACEMENT"
_U = ("Ux", "Uy", "Uz")


@dataclass(frozen=True, eq=False)
class CaseQuantity:
    """One result of a case: ``data`` is ``(S, n, c)``; ``ids`` for nodes only."""

    label: str
    target: str
    components: tuple[str, ...]
    data: np.ndarray
    ids: Optional[np.ndarray] = None


@dataclass(frozen=True, eq=False)
class SpectrumCase:
    """``/spectra/s<k>``: one spectrum along one direction (1-based)."""

    name: str
    direction: int
    scale: float
    spectrum_periods: np.ndarray
    spectrum_sa: np.ndarray
    mode: np.ndarray
    kind: tuple[str, ...]
    lam: np.ndarray
    sa_period: np.ndarray
    gamma: np.ndarray
    sa: np.ndarray
    damping: np.ndarray
    results: tuple[CaseQuantity, ...]
    reference: tuple[CaseQuantity, ...]

    def result(self, label: str) -> CaseQuantity:
        """The per-mode result ``label`` (``base_shear``, ``DISPLACEMENT``, ...)."""
        for q in self.results:
            if q.label == label:
                return q
        raise KeyError(f"case {self.name!r} holds no result {label!r}: "
                       f"{[q.label for q in self.results]}")


@dataclass(frozen=True)
class SpectraIndex:
    """The ``/spectra`` cases in force, plus each refused one."""

    cases: tuple[SpectrumCase, ...]
    refused: tuple[Refusal, ...]

    def by_name(self, name: str) -> SpectrumCase:
        for case in self.cases:
            if case.name == name:
                return case
        for ref in self.refused:
            if ref.name == name:
                raise KeyError(f"spectrum case {name!r} is refused: {ref.reason} ({ref.detail})")
        raise KeyError(f"no spectrum case named {name!r}")


# ---------------------------------------------------------------------------
# From the driver's result
# ---------------------------------------------------------------------------


def _nodal_layout(labels: Optional[Sequence[str]]) -> "Optional[tuple[str, list[int], list[int]]]":
    """``(kind, nodes, dofs)`` when ``labels`` are a node-major nodal extractor's."""
    if not labels:
        return None
    for kind, pattern in (("disp", _DISP_LABEL), ("reaction", _REACTION_LABEL)):
        found = [pattern.match(lab) for lab in labels]
        if not all(found):
            continue
        pairs = [(int(m.group(1)), int(m.group(2))) for m in found if m is not None]
        nodes = list(dict.fromkeys(n for n, _ in pairs))
        dofs = [d for n, d in pairs if n == nodes[0]]
        if pairs != [(n, d) for n in nodes for d in dofs]:
            return None  # not node-major over one DOF set: not a field
        return kind, nodes, dofs
    return None


def _damping_column(damping: "float | Sequence[float] | np.ndarray | None", m: int) -> np.ndarray:
    if damping is None:
        return np.full(m, np.nan)
    a = np.broadcast_to(np.asarray(damping, dtype="f8"), (m,)).astype("f8")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0) or np.any(a >= 1.0):
        raise ValueError(f"spectrum_cases: damping {a.tolist()} must lie in (0, 1)")
    return a


def spectrum_cases(
    result: SpectrumModalResult,
    names: Sequence[str],
    *,
    damping: "float | Sequence[float] | np.ndarray | None" = None,
    nodal: Sequence[str] = (),
    omit: Sequence[str] = (),
) -> tuple[SpectrumCase, ...]:
    """One :class:`SpectrumCase` per spectrum of ``result``, named by ``names``.

    ``result`` is what ``apeSees.spectrum_after_gravity`` returned. ``damping``
    (one ratio or one per mode, in ``(0, 1)``) is the case's per-mode column,
    the damping a ``cqc`` combination over the whole case uses; ``None`` writes
    NaN, which such a combination refuses. ``nodal`` names the ``NodeDisp``
    extractors to write as ``ON_NODES/DISPLACEMENT``, ``omit`` the extractors
    to leave out. A nodal extractor in neither is refused, and so is an
    unlabelled one with more than one value (module docstring).
    """
    if not isinstance(result, SpectrumModalResult):
        raise TypeError(
            f"spectrum_cases: expected a SpectrumModalResult, got {type(result).__name__}")
    spectra = tuple(result.spectra)
    names = tuple(names)
    if len(names) != len(spectra):
        raise ValueError(f"spectrum_cases: {len(names)} names for {len(spectra)} spectra")
    for name in names:
        problem = name_problem(name)
        if problem is not None:
            raise ValueError(f"spectrum_cases: case name {name!r}: {problem}")
    if len(set(names)) != len(names):
        raise ValueError(f"spectrum_cases: case names {list(names)} repeat")
    known = set(result.per_mode)
    for what, given in (("nodal", nodal), ("omit", omit)):
        unknown = sorted(set(given) - known)
        if unknown:
            raise ValueError(f"spectrum_cases: {what}= names {unknown}, which are not "
                             f"extractors of this result: {sorted(known)}")
    both = sorted(set(nodal) & set(omit))
    if both:
        raise ValueError(f"spectrum_cases: {both} are in both nodal= and omit=")

    lam = np.asarray(result.modal.eigenvalues, dtype="f8")
    m = lam.size
    periods = 2.0 * np.pi / np.sqrt(lam)  # the driver's own expression
    sa_period = (np.zeros(m) if result.sa_periods is None
                 else np.asarray(result.sa_periods, dtype="f8").copy())
    sa_at = np.where(sa_period > 0.0, sa_period, periods)
    kinds = tuple(result.kinds)
    props = result.modal.modal.properties
    ndm = int(np.asarray(result.per_mode[BASE_SHEAR]).shape[2])
    damp = _damping_column(damping, m)
    layout = _quantity_layout(result, nodal, omit, ndm)

    cases: list[SpectrumCase] = []
    for k, (name, s) in enumerate(zip(names, spectra)):
        if s.function is None:
            table_t = np.asarray(s.periods, dtype="f8")
            table_sa = np.asarray(s.accels, dtype="f8")
        else:
            table_t = np.unique(sa_at)
            table_sa = np.asarray(s.function(table_t), dtype="f8")
        token = ("MX", "MY", "MZ")[s.direction - 1]
        results_k: list[CaseQuantity] = []
        reference_k: list[CaseQuantity] = []
        for extractor, label, target, components, ids, shape in layout:
            per_mode = np.asarray(result.per_mode[extractor], dtype="f8")[k]
            ref = np.asarray(result.gravity_values[extractor], dtype="f8")
            results_k.append(CaseQuantity(label, target, components,
                                          per_mode.reshape((m,) + shape), ids))
            reference_k.append(CaseQuantity(label, target, components,
                                            ref.reshape((1,) + shape), ids))
        cases.append(SpectrumCase(
            name=name, direction=int(s.direction), scale=float(s.scale),
            spectrum_periods=table_t, spectrum_sa=table_sa,
            mode=np.arange(1, m + 1, dtype="i4"), kind=kinds, lam=lam.copy(),
            sa_period=sa_period, gamma=np.asarray(props[f"partiFactor{token}"], dtype="f8"),
            sa=np.asarray(s.sa(sa_at), dtype="f8"), damping=damp,
            results=tuple(results_k), reference=tuple(reference_k),
        ))
    for case in cases:
        problem = case_problem(case)
        if problem is not None:
            raise ValueError(f"spectrum_cases: case {case.name!r}: {problem}")
    return tuple(cases)


def _quantity_layout(
    result: SpectrumModalResult, nodal: Sequence[str], omit: Sequence[str], ndm: int,
) -> list[tuple[str, str, str, tuple[str, ...], Optional[np.ndarray], tuple[int, int]]]:
    """``(extractor, label, target, components, ids, (n, c))`` per written quantity."""
    out: list[tuple[str, str, str, tuple[str, ...], Optional[np.ndarray], tuple[int, int]]] = []
    for extractor, values in result.per_mode.items():
        if extractor in omit:
            continue
        width = int(np.asarray(values).shape[2])
        labels = result.labels.get(extractor)
        field = _nodal_layout(labels)
        if extractor in nodal:
            if field is None or field[0] != "disp":
                raise ValueError(
                    f"spectrum_cases: nodal= names {extractor!r}, whose labels are not a "
                    "NodeDisp's; only NodeDisp extractors are written as nodal fields")
            _, nodes, dofs = field
            if any(not 1 <= d <= ndm for d in dofs):
                raise ValueError(
                    f"spectrum_cases: {extractor!r} reads DOFs {dofs}; a nodal field here is "
                    f"the translational displacement, DOFs 1..{ndm}")
            out.append((extractor, _DISPLACEMENT, "nodes", tuple(_U[d - 1] for d in dofs),
                        np.asarray(nodes, dtype="i4"), (len(nodes), len(dofs))))
            continue
        if field is not None:
            raise ValueError(
                f"spectrum_cases: {extractor!r} holds nodal values "
                f"({len(field[1])} nodes); name it in nodal= to write the field or in "
                "omit= to leave it out (A7: nodal fields are written on request)")
        if not labels and width > 1 and extractor != BASE_SHEAR:
            raise ValueError(
                f"spectrum_cases: {extractor!r} returns {width} values with no labels (an "
                "ElementResponse, say); written to the domain they would lose which element "
                "and which force each value is. Give the extractor labels, or name it in "
                "omit= to leave it out")
        components = tuple(labels) if labels else tuple(str(i + 1) for i in range(width))
        out.append((extractor, extractor, "domain", components, None, (1, width)))
    labels_written = [entry[1] for entry in out]
    if len(set(labels_written)) != len(labels_written):
        raise ValueError(f"spectrum_cases: two quantities would be written as one result: "
                         f"{labels_written}")
    if not out:
        raise ValueError("spectrum_cases: omit= leaves no quantity to write")
    return out


# ---------------------------------------------------------------------------
# The rules, shared by the writer and the reader
# ---------------------------------------------------------------------------


def _quantities_problem(qs: Sequence[CaseQuantity], s: int, what: str) -> Optional[str]:
    if not qs:
        return f"{what} holds no result"
    seen: set[tuple[str, str]] = set()
    for q in qs:
        if q.target not in _ON:
            return f"{what} {q.label!r}: target {q.target!r} is not one of {sorted(_ON)}"
        problem = _link_problem(q.label)
        if problem is not None:
            return f"{what} label {q.label!r}: {problem}"
        if (q.target, q.label) in seen:
            return f"{what} holds {q.label!r} twice"
        seen.add((q.target, q.label))
        try:
            _token_problem(q.components, f"{what} {q.label} component")
        except EntryRefused as exc:
            return exc.detail
        data = np.asarray(q.data)
        if data.ndim != 3 or data.shape[0] != s or data.shape[2] != len(q.components):
            return (f"{what} {q.label!r} has shape {data.shape}, expected "
                    f"({s}, n, {len(q.components)})")
        if q.target == "domain" and data.shape[1] != 1:
            return f"{what} {q.label!r}: a domain value has n = 1, got {data.shape[1]}"
        if q.target == "nodes":
            ids = np.asarray(q.ids) if q.ids is not None else None
            if ids is None or ids.dtype.kind not in "iu" or ids.shape != (data.shape[1],):
                return f"{what} {q.label!r}: ids must be {data.shape[1]} integers"
    return None


def case_problem(c: SpectrumCase) -> Optional[str]:
    """Why ``c`` cannot be a ``/spectra`` case, or ``None``."""
    problem = name_problem(c.name)
    if problem is not None:
        return f"name {c.name!r}: {problem}"
    if c.direction < 1:
        return f"direction {c.direction} is 1-based"
    if not (math.isfinite(c.scale) and c.scale > 0.0):
        return f"scale {c.scale!r} must be finite and > 0"
    t = np.asarray(c.spectrum_periods, dtype="f8")
    sa_table = np.asarray(c.spectrum_sa, dtype="f8")
    if t.ndim != 1 or t.size == 0 or sa_table.shape != t.shape:
        return f"spectrum T {t.shape} and Sa {sa_table.shape} must be equal, non-empty columns"
    if np.any(t < 0.0) or np.any(np.diff(t) <= 0.0) or not np.all(np.isfinite(t)):
        return "spectrum T must be non-negative and strictly increasing"
    if not np.all(np.isfinite(sa_table)) or np.any(sa_table < 0.0):
        return "spectrum Sa must be finite and non-negative"
    mode = np.asarray(c.mode)
    m = mode.size
    if mode.ndim != 1 or m == 0 or mode.dtype.kind not in "iu":
        return f"modes/mode must be integers, got {mode.dtype} {mode.shape}"
    if int(mode.min()) < 1 or int(mode.max()) > 2 ** 31 - 1 or len(set(mode.tolist())) != m:
        return "modes/mode must be distinct, 1-based and fit i4"
    if len(c.kind) != m or any(k not in KINDS.values() for k in c.kind):
        return f"modes/kind must be one of {sorted(KINDS.values())} per mode"
    for what, col in (("lambda", c.lam), ("sa_period", c.sa_period), ("gamma", c.gamma),
                      ("sa", c.sa), ("damping", c.damping)):
        if np.asarray(col).shape != (m,):
            return f"modes/{what} has shape {np.asarray(col).shape}, expected ({m},)"
    lam = np.asarray(c.lam, dtype="f8")
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        return "every lambda must be finite and > 0 (lambda <= 0 is refused)"
    sa_period = np.asarray(c.sa_period, dtype="f8")
    if not np.all(np.isfinite(sa_period)) or np.any(sa_period < 0.0):
        return "sa_period must be finite and >= 0 (0 = the mode's own period)"
    if not np.all(np.isfinite(np.asarray(c.gamma, dtype="f8"))):
        return "gamma must be finite"
    sa = np.asarray(c.sa, dtype="f8")
    if not np.all(np.isfinite(sa)) or np.any(sa < 0.0):
        return "sa must be finite and non-negative"
    damping = np.asarray(c.damping, dtype="f8")
    finite = damping[~np.isnan(damping)]
    if np.any(np.isinf(damping)) or np.any(finite <= 0.0) or np.any(finite >= 1.0):
        return "damping must be NaN or lie in (0, 1)"
    for qs, s, what in ((c.results, m, "RESULTS"), (c.reference, 1, "reference/RESULTS")):
        problem = _quantities_problem(qs, s, what)
        if problem is not None:
            return problem
    shape_of = {(q.target, q.label): (np.asarray(q.data).shape[1:], q.components,
                                      None if q.ids is None else np.asarray(q.ids).tolist())
                for q in c.results}
    ref_of = {(q.target, q.label): (np.asarray(q.data).shape[1:], q.components,
                                    None if q.ids is None else np.asarray(q.ids).tolist())
              for q in c.reference}
    if shape_of != ref_of:
        return "reference/RESULTS must hold the results of RESULTS, with their ids and columns"
    return None


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------


def _write_quantities(root: h5py.Group, qs: Sequence[CaseQuantity], s: int) -> None:
    results = root.create_group("RESULTS")
    for q in qs:
        holder = results.require_group(_ON[q.target]).create_group(q.label)
        holder.attrs["COMPONENTS"] = ",".join(q.components)
        n = None
        if q.target == "nodes":
            n = _write_id(holder, q.ids, f"{q.label} ids")
        holder.create_dataset("DATA", data=_f8(q.data, f"{q.label} DATA", s, n, len(q.components)))
        _stack_axes(holder, s)


def write_spectra(out: h5py.File, cases: Sequence[SpectrumCase]) -> None:
    """Write ``/spectra`` into ``out``, which holds none. The caller checked ``cases``."""
    zone = out.create_group("spectra")
    for k, c in enumerate(cases):
        g = zone.create_group(f"s{k}")
        g.attrs["name"] = c.name
        g.attrs["KIND"] = "static"
        g.attrs["direction"] = np.int32(c.direction)
        g.attrs["scale"] = np.float64(c.scale)
        spec = g.create_group("spectrum")
        spec.create_dataset("T", data=np.asarray(c.spectrum_periods, dtype="f8"))
        spec.create_dataset("Sa", data=np.asarray(c.spectrum_sa, dtype="f8"))
        modes = g.create_group("modes")
        modes.create_dataset("mode", data=np.asarray(c.mode, dtype="i4"))
        modes.create_dataset("kind", data=list(c.kind), dtype=_STR)
        for name, col in (("lambda", c.lam), ("sa_period", c.sa_period), ("gamma", c.gamma),
                          ("sa", c.sa), ("damping", c.damping)):
            modes.create_dataset(name, data=np.asarray(col, dtype="f8"))
        _write_quantities(g, c.results, int(np.asarray(c.mode).size))
        _write_quantities(g.create_group("reference"), c.reference, 1)


def _attr(group: h5py.Group, key: str) -> Any:
    if key not in group.attrs:
        raise EntryRefused("bad-param", f"{group.name} has no @{key}")
    raw = group.attrs[key]
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise EntryRefused("bad-param", f"{group.name}@{key} holds {raw.size} values")
        raw = raw.ravel()[0]
    return raw


def _number(group: h5py.Group, key: str, kind: str) -> Any:
    raw = np.asarray(_attr(group, key))
    if raw.dtype.kind not in kind:
        raise EntryRefused("bad-param", f"{group.name}@{key} is {raw.dtype}, expected {kind}")
    return raw.item()


def _column(group: h5py.Group, name: str, kind: str) -> np.ndarray:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise EntryRefused("bad-param", f"{group.name} has no {name} column")
    ds = group[name]
    good = h5py.check_string_dtype(ds.dtype) is not None if kind == "str" else (
        (ds.dtype.kind, ds.dtype.itemsize) == (kind[0], int(kind[1:])))
    if not good or ds.ndim != 1:
        raise EntryRefused("bad-param", f"{ds.name} is {ds.dtype} {ds.shape}, expected {kind}")
    return ds[...]


def _read_quantities(root: h5py.Group, where: str) -> tuple[CaseQuantity, ...]:
    if "RESULTS" not in root or not isinstance(root["RESULTS"], h5py.Group):
        raise EntryRefused("bad-param", f"{root.name} has no {where}")
    out: list[CaseQuantity] = []
    for on in root["RESULTS"]:
        if on not in _TARGET:
            raise EntryRefused(
                "bad-param", f"{root.name}/RESULTS/{on}: the family {on} is not in spectra "
                f"{SPECTRA_SCHEMA_VERSION}, whose cases hold {sorted(_TARGET)} only")
        family = root["RESULTS"][on]
        if not isinstance(family, h5py.Group):
            raise EntryRefused("bad-param", f"{family.name} is not a group")
        for label in family:
            holder = family[label]
            if not isinstance(holder, h5py.Group) or "DATA" not in holder:
                raise EntryRefused("bad-param", f"{holder.name} is not a result group")
            data = holder["DATA"]
            if not isinstance(data, h5py.Dataset) or data.dtype != np.dtype("f8") or data.ndim != 3:
                raise EntryRefused("bad-param", f"{data.name} is not f8 of rank 3")
            s = data.shape[0]
            for axis in ("STEP", "TIME"):
                if axis not in holder or not isinstance(holder[axis], h5py.Dataset) or \
                        holder[axis].shape != (s,):
                    raise EntryRefused("bad-param", f"{holder.name}: {axis} must have {s} rows")
            ids = None
            if on == "ON_NODES":
                found = holder["ID"] if "ID" in holder else None
                if not isinstance(found, h5py.Dataset) or found.dtype != np.dtype("i4") or \
                        found.shape != (data.shape[1], 1):
                    raise EntryRefused("bad-param", f"{holder.name}/ID must be i4 ({data.shape[1]}, 1)")
                ids = found[...].ravel()
            components = tuple(text_or_empty(_attr(holder, "COMPONENTS")).split(","))
            out.append(CaseQuantity(label, _TARGET[on], components, data[...], ids))
    return tuple(out)


def read_case(group: h5py.Group) -> SpectrumCase:
    """One case, validated by :func:`case_problem`, or :class:`EntryRefused`."""
    name = text_or_empty(_attr(group, "name"))
    problem = name_problem(name)
    if problem is not None:
        raise EntryRefused("empty-name", f"{group.name}@name: {problem}")
    kind_attr = text_or_empty(_attr(group, "KIND"))
    if kind_attr != "static":
        raise EntryRefused("bad-param", f"{group.name}@KIND is {kind_attr!r}, not 'static'")
    if "spectrum" not in group or "modes" not in group or "reference" not in group:
        raise EntryRefused("bad-param", f"{group.name} needs spectrum, modes and reference")
    modes = group["modes"]
    case = SpectrumCase(
        name=name,
        direction=int(_number(group, "direction", "iu")),
        scale=float(_number(group, "scale", "f")),
        spectrum_periods=_column(group["spectrum"], "T", "f8"),
        spectrum_sa=_column(group["spectrum"], "Sa", "f8"),
        mode=_column(modes, "mode", "i4"),
        kind=tuple(decode_cell(k, "/spectra", f"{modes.name}/kind")
                   for k in _column(modes, "kind", "str")),
        lam=_column(modes, "lambda", "f8"),
        sa_period=_column(modes, "sa_period", "f8"),
        gamma=_column(modes, "gamma", "f8"),
        sa=_column(modes, "sa", "f8"),
        damping=_column(modes, "damping", "f8"),
        results=_read_quantities(group, "RESULTS"),
        reference=_read_quantities(group["reference"], "reference/RESULTS"),
    )
    problem = case_problem(case)
    if problem is not None:
        raise EntryRefused("bad-param", f"{group.name}: {problem}")
    return case


def _case_name(group: Any) -> str:
    """A case's ``@name`` for a refusal, ``""`` when even that cannot be read."""
    if isinstance(group, h5py.Group) and "name" in group.attrs:
        return text_or_empty(group.attrs["name"])
    return ""


def read_spectra(zone: h5py.Group) -> SpectraIndex:
    """Every ``s<k>`` case, a bad one refused alone, duplicate names both refused."""
    links = sorted((k for k in zone if _CASE.match(k)), key=lambda k: int(k[1:]))
    cases: list[tuple[int, SpectrumCase]] = []
    refused: list[Refusal] = []
    for at, link in enumerate(links):
        group = zone[link]
        try:
            if not isinstance(group, h5py.Group):
                raise EntryRefused("bad-param", f"{group.name} is not a group")
            cases.append((at, read_case(group)))
        except ZoneRefused:  # a non-UTF-8 table cell refuses the zone
            raise
        except EntryRefused as exc:
            refused.append(Refusal(_case_name(group), exc.reason, exc.detail, row=at))
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            # What h5py raises on a damaged case (a broken link, a dataset where a
            # group belongs) refuses that case; the others still read.
            refused.append(Refusal(_case_name(group), "bad-param", str(exc), row=at))
    counts: dict[str, int] = {}
    for _, case in cases:
        counts[case.name] = counts.get(case.name, 0) + 1
    for at, case in cases:
        if counts[case.name] > 1:
            refused.append(Refusal(case.name, "duplicate",
                                   f"{counts[case.name]} cases are named {case.name!r}", row=at))
    return SpectraIndex(tuple(c for _, c in cases if counts[c.name] == 1), tuple(refused))


def same_bits(a: SpectrumCase, b: SpectrumCase) -> bool:
    """Every value of two cases equal bit for bit (NaN included)."""
    def bits(x: Any, dtype: str) -> bytes:
        return np.asarray(x, dtype=dtype).tobytes()

    if (a.name, a.direction, a.kind) != (b.name, b.direction, b.kind):
        return False
    if bits(a.scale, "f8") != bits(b.scale, "f8"):
        return False
    for x, y in ((a.spectrum_periods, b.spectrum_periods), (a.spectrum_sa, b.spectrum_sa),
                 (a.lam, b.lam), (a.sa_period, b.sa_period), (a.gamma, b.gamma),
                 (a.sa, b.sa), (a.damping, b.damping)):
        if np.asarray(x).shape != np.asarray(y).shape or bits(x, "f8") != bits(y, "f8"):
            return False
    if bits(a.mode, "i4") != bits(b.mode, "i4"):
        return False
    for xs, ys in ((a.results, b.results), (a.reference, b.reference)):
        mine = {(q.target, q.label): q for q in xs}
        theirs = {(q.target, q.label): q for q in ys}
        if set(mine) != set(theirs):
            return False
        for key, q in mine.items():
            r = theirs[key]
            if q.components != r.components or np.asarray(q.data).shape != np.asarray(r.data).shape:
                return False
            if bits(q.data, "f8") != bits(r.data, "f8"):
                return False
            if (q.ids is None) != (r.ids is None) or (
                    q.ids is not None and bits(q.ids, "i4") != bits(r.ids, "i4")):
                return False
    return True
