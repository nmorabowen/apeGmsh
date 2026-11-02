"""The ``/modes`` table: the modal properties the engineer reads first (ADR 0130 D6).

One row per mode. ``host`` and ``step`` name the modal set in the paired
``.ladruno`` (a stage's ``source`` and its ``STEP_<s>`` id, ``-1`` for the only
set), ``mode`` is 1-based. The ratios are fractions, not the percent that
``modalProperties`` returns.

**Generalised mass** (A1). ``modalProperties`` does not export it, so it is
stored as the effective mass over ``Γ²`` on the component of largest ``|Γ|``,
and NaN where every ``Γ`` is zero. With ``Γ_c = L_c / M*`` and
``M_eff,c = L_c² / M*`` the quotient is ``M* = φᵀ M φ`` on every component with
``Γ_c ≠ 0``; the largest ``|Γ|`` is the one with the least round-off.

**A row is stale** (A6) when its ``lambda`` differs from the ``.ladruno``'s
``LAMBDA`` of the same mode by more than ``1e-9`` relative. A row the
``.ladruno`` does not hold, or that cannot be matched to it, is stale too:
nothing is drawn as current on no proof.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from ..readers._ladruno_multi import discover_partition_files
from ..readers._mpco_nodal_io import discover_modes
from ._descriptor import decode_cell
from ._errors import ZoneRefused
from ._names import name_problem, text_or_empty

__all__ = ["COMPONENTS", "LAMBDA_RTOL", "ModeTable", "ladruno_lambdas", "mode_table"]

#: The component tokens ``modalProperties`` uses, in the order a table lists them.
COMPONENTS = ("MX", "MY", "MZ", "RMX", "RMY", "RMZ")
#: ``modeKind`` of ``modalProperties -return`` (fork ADR 213), by tag.
KINDS = {0: "eigen", 1: "residual", 2: "ritz"}
#: A6: the relative ``lambda`` gap past which a row is stale.
LAMBDA_RTOL = 1e-9
_ZONE = "/modes"
_STR = h5py.string_dtype(encoding="utf-8")
_I4_MIN, _I4_MAX = -(2 ** 31), 2 ** 31 - 1
_HOST = re.compile(r"stage_(\d+)\Z")
_TABLE_COLUMNS = ("participation", "mass_ratio", "cumulative_ratio")


@dataclass(frozen=True, eq=False)
class ModeTable:
    """``/modes``: ``M`` rows, ``C`` components. Every array is a copy the table owns."""

    components: tuple[str, ...]
    host: tuple[str, ...]
    kind: tuple[str, ...]
    step: np.ndarray
    mode: np.ndarray
    lam: np.ndarray
    period: np.ndarray
    generalized_mass: np.ndarray
    participation: np.ndarray
    mass_ratio: np.ndarray
    cumulative_ratio: np.ndarray

    def __len__(self) -> int:
        return len(self.host)

    def same_bits(self, other: "ModeTable") -> bool:
        """Every column equal bit for bit (NaN included), the strings byte-equal."""
        if (self.components, self.host, self.kind) != (other.components, other.host, other.kind):
            return False
        pairs = (
            (self.step, other.step, "i4"), (self.mode, other.mode, "i4"),
            (self.lam, other.lam, "f8"), (self.period, other.period, "f8"),
            (self.generalized_mass, other.generalized_mass, "f8"),
            (self.participation, other.participation, "f8"),
            (self.mass_ratio, other.mass_ratio, "f8"),
            (self.cumulative_ratio, other.cumulative_ratio, "f8"),
        )
        return all(
            np.asarray(a).shape == np.asarray(b).shape
            and np.asarray(a, dtype=t).tobytes() == np.asarray(b, dtype=t).tobytes()
            for a, b, t in pairs
        )


# ---------------------------------------------------------------------------
# From modalProperties
# ---------------------------------------------------------------------------


def _series(properties: Mapping[str, Any], key: str, n: Optional[int]) -> np.ndarray:
    if key not in properties:
        raise ValueError(
            f"mode_table: modalProperties has no {key!r}; available: {sorted(properties)}")
    a = np.asarray(properties[key], dtype="f8").ravel()
    if n is not None and a.size != n:
        raise ValueError(f"mode_table: {key!r} has {a.size} values for {n} modes")
    return a


def mode_table(
    properties: Mapping[str, Any],
    *,
    host: str,
    step: int = -1,
) -> ModeTable:
    """The ``/modes`` table of one ``modalProperties -return`` dict.

    ``properties`` is :attr:`ModalPropertiesResult.properties
    <apeGmsh.opensees.analysis.modal.ModalPropertiesResult.properties>` (also
    reached as ``GravityModalResult.modal.properties``). ``host`` is the stage
    whose modal set these modes are, as a source address names it, and
    ``step`` its ``STEP_<s>`` id (``-1``: the stage's only set).

    ``lambda`` and ``period`` are ``eigenLambda`` and ``eigenPeriod`` as
    returned, the ratios are ``partiMassRatios<C>`` and ``partiMassRatiosCumu<C>``
    over 100, and ``participation`` is ``partiFactor<C>``, for every component
    ``C`` the dict carries. Raises ``ValueError`` on a dict that lacks one of
    these or a ``modeKind`` it does not know.
    """
    lam = _series(properties, "eigenLambda", None)
    n = lam.size
    period = _series(properties, "eigenPeriod", n)
    comps = tuple(c for c in COMPONENTS if f"partiFactor{c}" in properties)
    if not comps:
        raise ValueError(
            "mode_table: modalProperties carries no partiFactor<C>; "
            f"available: {sorted(properties)}")
    gamma = np.stack([_series(properties, f"partiFactor{c}", n) for c in comps], axis=1)
    meff = np.stack([_series(properties, f"partiMass{c}", n) for c in comps], axis=1)
    ratio = np.stack(
        [_series(properties, f"partiMassRatios{c}", n) for c in comps], axis=1) / 100.0
    cumu = np.stack(
        [_series(properties, f"partiMassRatiosCumu{c}", n) for c in comps], axis=1) / 100.0
    gm = np.full(n, np.nan)
    for i in range(n):
        a = int(np.argmax(np.abs(gamma[i])))
        if gamma[i, a] != 0.0:
            gm[i] = meff[i, a] / (gamma[i, a] * gamma[i, a])
    raw = properties.get("modeKind")
    if raw is None:
        kinds = ("eigen",) * n
    else:
        tags = [int(k) for k in np.asarray(raw).ravel()]
        if len(tags) != n or any(t not in KINDS for t in tags):
            raise ValueError(
                f"mode_table: modeKind {tags} is not one of {sorted(KINDS)} per mode")
        kinds = tuple(KINDS[t] for t in tags)
    table = ModeTable(
        components=comps, host=(host,) * n, kind=kinds,
        step=np.full(n, int(step), dtype="i4"), mode=np.arange(1, n + 1, dtype="i4"),
        lam=lam, period=period, generalized_mass=gm,
        participation=gamma, mass_ratio=ratio, cumulative_ratio=cumu,
    )
    problem = table_problem(table)
    if problem is not None:
        raise ValueError(f"mode_table: {problem}")
    return table


# ---------------------------------------------------------------------------
# The rules, shared by the writer and the reader
# ---------------------------------------------------------------------------


def table_problem(t: ModeTable) -> Optional[str]:
    """Why ``t`` cannot be ``/modes``, or ``None``."""
    m, c = len(t.host), len(t.components)
    if m == 0:
        return "the table has no rows"
    if c == 0 or len(set(t.components)) != c or any(x not in COMPONENTS for x in t.components):
        return f"components {t.components} must be distinct tokens of {COMPONENTS}"
    if len(t.kind) != m:
        return f"{len(t.kind)} kinds for {m} rows"
    for host in t.host:
        problem = name_problem(host)
        if problem is not None:
            return f"host {host!r}: {problem}"
    bad = sorted(set(t.kind) - set(KINDS.values()))
    if bad:
        return f"kinds {bad} are not one of {sorted(KINDS.values())}"
    for what, col in (("step", t.step), ("mode", t.mode)):
        a = np.asarray(col)
        if a.shape != (m,) or a.dtype.kind not in "iu":
            return f"{what} must be {m} integers, got {a.dtype} {a.shape}"
        if a.size and (int(a.min()) < _I4_MIN or int(a.max()) > _I4_MAX):
            return f"{what} does not fit i4"
    if int(np.min(t.step)) < -1:
        return "a step is below -1"
    if int(np.min(t.mode)) < 1:
        return "modes are 1-based"
    keys = list(zip(t.host, (int(s) for s in t.step), (int(k) for k in t.mode)))
    if len(set(keys)) != m:
        return "two rows name the same (host, step, mode)"
    for what, col in (("lambda", t.lam), ("period", t.period),
                      ("generalized_mass", t.generalized_mass)):
        a = np.asarray(col, dtype="f8")
        if a.shape != (m,):
            return f"{what} has shape {a.shape}, expected ({m},)"
    lam = np.asarray(t.lam, dtype="f8")
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        return "every lambda must be finite and > 0 (lambda <= 0 is refused)"
    period = np.asarray(t.period, dtype="f8")
    if not np.all(np.isfinite(period)) or np.any(period <= 0.0):
        return "every period must be finite and > 0"
    if np.any(np.isinf(np.asarray(t.generalized_mass, dtype="f8"))):
        return "a generalized_mass is infinite"
    for what, col in (("participation", t.participation), ("mass_ratio", t.mass_ratio),
                      ("cumulative_ratio", t.cumulative_ratio)):
        a = np.asarray(col, dtype="f8")
        if a.shape != (m, c):
            return f"{what} has shape {a.shape}, expected ({m}, {c})"
        if np.any(np.isinf(a)):
            return f"{what} holds an infinite value"
    return None


def write_modes(out: h5py.File, t: ModeTable) -> None:
    """Write ``/modes`` into ``out``, which holds none. The caller checked ``t``."""
    zone = out.create_group("modes")
    zone.attrs["components"] = ",".join(t.components)
    zone.create_dataset("host", data=list(t.host), dtype=_STR)
    zone.create_dataset("kind", data=list(t.kind), dtype=_STR)
    zone.create_dataset("step", data=np.asarray(t.step, dtype="i4"))
    zone.create_dataset("mode", data=np.asarray(t.mode, dtype="i4"))
    zone.create_dataset("lambda", data=np.asarray(t.lam, dtype="f8"))
    zone.create_dataset("period", data=np.asarray(t.period, dtype="f8"))
    zone.create_dataset("generalized_mass", data=np.asarray(t.generalized_mass, dtype="f8"))
    for name in _TABLE_COLUMNS:
        zone.create_dataset(name, data=np.asarray(getattr(t, name), dtype="f8"))


def _col(zone: h5py.Group, name: str, kind: str, ndim: int) -> np.ndarray:
    if name not in zone or not isinstance(zone[name], h5py.Dataset):
        raise ZoneRefused(_ZONE, f"it has no {name} column")
    ds = zone[name]
    good = h5py.check_string_dtype(ds.dtype) is not None if kind == "str" else (
        (ds.dtype.kind, ds.dtype.itemsize) == (kind[0], int(kind[1:])))
    if not good or ds.ndim != ndim:
        raise ZoneRefused(_ZONE, f"{name} is {ds.dtype} {ds.shape}, expected {kind}, rank {ndim}")
    return ds[...]


def read_modes(zone: h5py.Group) -> ModeTable:
    """``/modes`` as a :class:`ModeTable`. Malformed columns refuse the zone."""
    if not isinstance(zone, h5py.Group):
        raise ZoneRefused(_ZONE, "it is not a group")
    if "components" not in zone.attrs:
        raise ZoneRefused(_ZONE, "it has no @components")
    raw = zone.attrs["components"]
    if isinstance(raw, np.ndarray):
        raw = raw.ravel()[0] if raw.size == 1 else ""
    components = tuple(text_or_empty(raw).split(","))
    table = ModeTable(
        components=components,
        host=tuple(decode_cell(x, _ZONE, "host") for x in _col(zone, "host", "str", 1)),
        kind=tuple(decode_cell(x, _ZONE, "kind") for x in _col(zone, "kind", "str", 1)),
        step=_col(zone, "step", "i4", 1),
        mode=_col(zone, "mode", "i4", 1),
        lam=_col(zone, "lambda", "f8", 1),
        period=_col(zone, "period", "f8", 1),
        generalized_mass=_col(zone, "generalized_mass", "f8", 1),
        participation=_col(zone, "participation", "f8", 2),
        mass_ratio=_col(zone, "mass_ratio", "f8", 2),
        cumulative_ratio=_col(zone, "cumulative_ratio", "f8", 2),
    )
    problem = table_problem(table)
    if problem is not None:
        raise ZoneRefused(_ZONE, problem)
    return table


# ---------------------------------------------------------------------------
# The row rule of A6
# ---------------------------------------------------------------------------


def ladruno_lambdas(results_file: "str | Path") -> dict[tuple[str, int], dict[int, float]]:
    """``{(host, s): {mode: LAMBDA}}`` of every modal set a ``.ladruno`` records.

    ``host`` is ``stage_<i>``, the ``i``-th ``MODEL_STAGE`` by its index, which is
    how a stage is named when the run stamped none; ``s`` is the ``STEP_<s>`` id.
    A part set is read from its first part: every part records the same modes.
    Raises ``ValueError`` on a file whose modal record is inconsistent.
    """
    path = Path(results_file)
    first = discover_partition_files(path)[0]
    out: dict[tuple[str, int], dict[int, float]] = {}
    with h5py.File(first, "r") as f:
        stages = sorted(
            (k for k in f if k.startswith("MODEL_STAGE[")),
            key=lambda k: int(k[len("MODEL_STAGE["):].rstrip("]")))
        for i, stage in enumerate(stages):
            group = f[stage]
            if "RESULTS" not in group or "ON_NODES" not in group["RESULTS"]:
                continue
            for step_key, entries in discover_modes(
                    group["RESULTS/ON_NODES"], source=f"{first.name} {stage}"):
                out[(f"stage_{i}", int(step_key[len("STEP_"):]))] = {
                    e.mode_index: e.eigenvalue for e in entries}
    return out


def row_staleness(
    t: ModeTable, lambdas: Mapping[tuple[str, int], Mapping[int, float]],
) -> tuple[tuple[bool, str], ...]:
    """``(stale, reason)`` per row of ``t`` against :func:`ladruno_lambdas`."""
    out: list[tuple[bool, str]] = []
    for host, step, mode, lam in zip(t.host, t.step, t.mode, t.lam):
        out.append(_one_row(host, int(step), int(mode), float(lam), lambdas))
    return tuple(out)


def _one_row(
    host: str, step: int, mode: int, lam: float,
    lambdas: Mapping[tuple[str, int], Mapping[int, float]],
) -> tuple[bool, str]:
    if _HOST.match(host) is None:
        return True, (f"host {host!r} is not a stage_<i> name; a stamped stage name "
                      "is not resolved against the .ladruno here")
    sets = sorted(s for (h, s) in lambdas if h == host)
    if not sets:
        return True, f"the .ladruno records no modes on {host}"
    if step == -1:
        if len(sets) != 1:
            return True, f"step -1 names the only set, and {host} holds {len(sets)}"
        step = sets[0]
    if (host, step) not in lambdas:
        return True, f"the .ladruno records no STEP_{step} on {host}"
    recorded = lambdas[(host, step)].get(mode)
    if recorded is None:
        return True, f"the .ladruno records no mode {mode} on {host} STEP_{step}"
    gap = abs(lam - recorded)
    if not math.isfinite(gap) or gap > LAMBDA_RTOL * abs(recorded):
        return True, (f"lambda {lam!r} here, {recorded!r} in the .ladruno "
                      f"(more than {LAMBDA_RTOL:g} relative)")
    return False, ""

