"""Write ``conformance.json``: the combination fixture both evaluators share (ADR 0130 D4).

The Python evaluator (``apeGmsh.results.user._evaluate``, slice VU2) and the app's
TypeScript evaluator (slice VU4) read the same inputs and must reproduce the same
goldens to ``1e-12`` relative. Run from the repository root::

    python tests/fixtures/user_results/make_conformance.py

The goldens are computed here by a reference written for this file alone: plain
Python loops over ``{(row key, label): value}`` maps, sharing no code with either
evaluator. ``tests/results/user/test_user_conformance.py`` checks the committed
file against this script, the evaluator against the goldens, and the goldens
against ``apeGmsh.opensees.analysis.spectrum`` (``combine_modes``,
``combine_directions``, ``seismic_envelope``).

JSON has no NaN, so a NaN is ``null``. A field is ``{"key_names", "keys",
"components", "values"}``; ``keys`` holds one list per row and ``values`` one list
per row, one value per component. An address is written ``kind|source|step|mode``.

What the file holds: the inputs (``states`` and ``other_states`` on Gauss points
and entities, ``eigenvalues``, ``cases``), the ``.user.h5`` rows
(``combinations``) and the ``model.h5`` rows (``model_combinations``), each
quantity's ``@linear`` flag and stored rows, and four kinds of golden: ``table``
(the verdict on every refused row), ``goldens`` (each combination per quantity),
and ``lone`` (one state read alone, with no rule). A refused evaluation is
``{"refused": <reason>}``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
OUT = HERE / "conformance.json"
NAN = math.nan

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

NODES = [[1], [2], [3]]


def nodal(rows: list[list[float]], components: tuple[str, ...] = ("ux", "uy"),
          keys: Optional[list[list[int]]] = None) -> dict[str, Any]:
    return {"key_names": ["id"], "keys": keys or NODES, "components": list(components),
            "values": rows}


def address(kind: str, source: str, step: int = -1, mode: int = 0) -> str:
    return f"{kind}|{source}|{step}|{mode}"


def omega_sq(period: float) -> float:
    return (2.0 * math.pi / period) ** 2


STATES = {
    address("stage", "D"): nodal([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    address("stage", "L"): nodal([[0.5, -1.0], [2.0, 0.0], [-3.0, 1.0]]),
    # W lacks node 3 and the uy label: the join reads NaN there.
    address("stage", "W"): nodal([[10.0], [20.0]], ("ux",), [[1], [2]]),
    # E holds a NaN cell, which every rule propagates (never nanmax).
    address("stage", "E"): nodal([[1.0, NAN], [-7.0, 2.0], [3.0, -3.0]]),
    address("mode", "eig", -1, 1): nodal([[1.0, 0.0], [0.5, 0.2], [0.0, 1.0]]),
    address("mode", "eig", -1, 2): nodal([[0.0, 1.0], [-0.5, 0.3], [1.0, 0.0]]),
    address("mode", "eig", -1, 3): nodal([[0.2, 0.2], [0.1, -0.4], [-0.3, 0.6]]),
    address("spectrum", "Ex", -1, 1): nodal([[3.0, 1.0], [2.0, -2.0], [1.0, 0.5]]),
    address("spectrum", "Ex", -1, 2): nodal([[4.0, -1.0], [-1.0, 2.0], [0.5, 1.0]]),
    address("reference", "Ex"): nodal([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]),
}

def keyed(key_names: list[str], keys: list[list[Any]], components: tuple[str, ...],
          rows: list[list[float]]) -> dict[str, Any]:
    return {"key_names": key_names, "keys": keys, "components": list(components), "values": rows}


#: States on other targets, read by the recorded quantity only. The Gauss terms
#: share element 5's second point and element 6's first; the entity terms share
#: the cut, and each has a row the other lacks.
OTHER_STATES = {
    address("stage", "GD"): keyed(["id", "sub"], [[5, 0], [5, 1], [6, 0]], ("sxx",),
                                  [[1.0], [2.0], [3.0]]),
    address("stage", "GL"): keyed(["id", "sub"], [[5, 1], [6, 0], [6, 1]], ("sxx",),
                                  [[10.0], [20.0], [30.0]]),
    address("stage", "XD"): keyed(["kind", "row"], [["cut", 0], ["storey", 2]], ("V",),
                                  [[4.0], [-1.0]]),
    address("stage", "XL"): keyed(["kind", "row"], [["cut", 0], ["a_point_set", 1]], ("V",),
                                  [[6.0], [2.5]]),
}

EIGENVALUES = {
    address("mode", "eig", -1, 1): omega_sq(1.0),
    address("mode", "eig", -1, 2): omega_sq(0.5),
    address("mode", "eig", -1, 3): omega_sq(0.2),
}


def case(periods: list[float], sa_period: list[float], damping: list[float],
         increments: list[list[list[float]]]) -> dict[str, Any]:
    return {
        "mode": list(range(1, len(periods) + 1)),
        "lambda": [omega_sq(t) for t in periods],
        "sa_period": sa_period,
        "damping": damping,
        "increments": [nodal(rows) for rows in increments],
    }


CASES = {
    # Modes 3 and 4 share sa_period 0.05: one rigid term, at omega = 2 pi / 0.05.
    "Ex": case(
        [1.0, 0.9, 0.08, 0.06], [0.0, 0.0, 0.05, 0.05], [0.05, 0.05, 0.03, 0.02],
        [[[3.0, 1.0], [2.0, -2.0], [1.0, 0.5]],
         [[4.0, -1.0], [-1.0, 2.0], [0.5, 1.0]],
         [[0.5, 0.25], [1.0, -0.5], [0.2, 0.1]],
         [[-0.25, 0.5], [0.5, 0.5], [0.1, -0.3]]],
    ),
    "Ey": case(
        [0.8, 0.3, 0.1], [0.0, 0.0, 0.0], [0.05, 0.05, 0.05],
        [[[1.0, 3.0], [-2.0, 1.0], [0.5, 2.0]],
         [[0.5, -1.0], [1.0, 1.0], [-0.5, 0.5]],
         [[0.1, 0.2], [0.3, -0.2], [0.2, 0.1]]],
    ),
    # A non-positive eigenvalue: CQC over this case refuses.
    "Ebad": {**case([1.0], [0.0], [0.05], [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]),
             "lambda": [0.0]},
}


def term(kind: str, source: str, factor: float = 1.0, step: int = -1, mode: int = 0) -> dict[str, Any]:
    return {"kind": kind, "source": source, "step": step, "mode": mode, "factor": factor}


def combo(name: str, rule: str, terms: list[dict[str, Any]], *, envelope: str = "",
          fraction: Optional[float] = None, damping: Optional[float] = None) -> dict[str, Any]:
    return {"name": name, "rule": rule, "terms": terms, "envelope": envelope,
            "fraction": fraction, "damping": damping}


S, M, C = "stage", "mode", "combination"
COMBINATIONS = [
    combo("ULS", "linear", [term(S, "D", 1.2), term(S, "L", 1.6)]),
    combo("D+W", "linear", [term(S, "D"), term(S, "W")]),
    combo("srss_list", "srss", [term(S, "D"), term(S, "L", -2.0)]),
    combo("abs_list", "abs_sum", [term(S, "D"), term(S, "L", -2.0)]),
    combo("env_max", "envelope", [term(S, "D"), term(S, "L", 2.0), term(S, "E")], envelope="max"),
    combo("env_min", "envelope", [term(S, "D"), term(S, "L", 2.0), term(S, "E")], envelope="min"),
    combo("env_absmax", "envelope", [term(S, "D", -1.0), term(S, "L", 2.0)], envelope="absmax"),
    combo("modal_cqc", "cqc", [term(M, "eig", 1.0, -1, k) for k in (1, 2, 3)], damping=0.05),
    combo("modal_srss", "srss", [term(M, "eig", 1.0, -1, k) for k in (1, 2, 3)]),
    combo("Ex_cqc", "cqc", [term("spectrum", "Ex")]),
    combo("Ex_cqc_d02", "cqc", [term("spectrum", "Ex")], damping=0.02),
    combo("Ex_srss", "srss", [term("spectrum", "Ex", 2.0)]),
    combo("Ex_abs", "abs_sum", [term("spectrum", "Ex")]),
    combo("Ey_cqc", "cqc", [term("spectrum", "Ey")]),
    combo("dir", "directional", [term(C, "Ex_cqc"), term(C, "Ey_cqc")], fraction=0.3),
    combo("plus", "linear", [term(C, "ULS"), term(C, "dir")]),
    combo("minus", "linear", [term(C, "ULS"), term(C, "dir", -1.0)]),
    combo("seismic_max", "envelope", [term(C, "plus"), term(C, "minus")], envelope="max"),
    combo("seismic_min", "envelope", [term(C, "plus"), term(C, "minus")], envelope="min"),
    combo("k_cqc", "cqc", [term("spectrum", "Ex", 1.0, -1, 1), term("spectrum", "Ex", -1.0, -1, 2)]),
    combo("k_linear", "linear", [term("spectrum", "Ex", 1.0, -1, 1), term("reference", "Ex", 2.0)]),
    combo("twice_ULS", "linear", [term(C, "ULS", 2.0)]),
    combo("bad_lambda", "cqc", [term("spectrum", "Ebad")]),
    combo("no_stage", "linear", [term(S, "Q")]),
    combo("env_unit", "envelope", [term(S, "D"), term(S, "L"), term(S, "E")], envelope="max"),
    combo("gauss_ULS", "linear", [term(S, "GD", 1.2), term(S, "GL", 1.6)]),
    combo("entity_env", "envelope", [term(S, "XD"), term(S, "XL")], envelope="max"),
    combo("uses_model", "linear", [term(C, "M_base", 1.5)]),
    combo("clash", "linear", [term(S, "D")]),
    combo("cyc_a", "linear", [term(C, "cyc_b")]),
    combo("cyc_b", "linear", [term(C, "cyc_a")]),
    combo("reaches_cyc", "linear", [term(S, "D"), term(C, "cyc_a")]),
    combo("on_absent", "linear", [term(C, "nobody")]),
]

#: Rows of ``model.h5`` ``/opensees/combinations``: one the user file names, and
#: one whose name the user file uses too, so both rows of that name refuse.
MODEL_COMBINATIONS = [
    combo("M_base", "linear", [term(S, "D", 2.0)]),
    combo("clash", "linear", [term(S, "L")]),
]

#: Each quantity: whether it is ``@linear``, and the combinations it stores a row for.
#: ``recorded`` reads ``STATES`` above; the user results scale them by their own
#: factor, so their values differ.
QUANTITIES = {
    "recorded": {"linear": True, "scale": 1.0, "stored": {}},
    "m_xx": {"linear": True, "scale": 0.5, "stored": {}},
    # wood_armer_top is not linear: only envelope, or a stored row, gives a value.
    "wood_armer_top": {
        "linear": False, "scale": 3.0,
        "stored": {"ULS": nodal([[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]])},
    },
}
#: The combinations goldened per quantity (every one for ``recorded``).
USER_COMBINATIONS = ["ULS", "env_max", "env_unit", "env_absmax", "srss_list", "twice_ULS",
                     "plus", "uses_model"]
#: One state read alone, per quantity: no rule, so no @linear check and no badge.
LONE = {
    "recorded": [address(S, "D"), address(S, "W"), address(M, "eig", -1, 2),
                 address("spectrum", "Ex", -1, 1), address("reference", "Ex"),
                 address(S, "GD"), address("spectrum", "Ex"), address(S, "Q")],
    "m_xx": [address(S, "D")],
    "wood_armer_top": [address(S, "D"), address(S, "L"), address(S, "Q")],
}


# ---------------------------------------------------------------------------
# The reference: loops over {(key, label): value}, sharing no evaluator code
# ---------------------------------------------------------------------------

Cells = dict[tuple[tuple[Any, ...], str], float]


def table_verdicts() -> dict[str, dict[str, str]]:
    """The verdict on each refused row (A2, A3), by a fixed point, not a peel."""
    rows = [("user", c) for c in COMBINATIONS] + [("model", c) for c in MODEL_COMBINATIONS]
    verdict: dict[int, str] = {}
    for i, (_, c) in enumerate(rows):
        same = [o for o, d in rows if d["name"] == c["name"]]
        if len(same) > 1:
            verdict[i] = "clash" if len(set(same)) > 1 else "duplicate"
    standing = {c["name"]: i for i, (_, c) in enumerate(rows) if i not in verdict}

    def targets(i: int) -> list[Any]:
        """Each combination row ``i`` names: its index, or None if not visible to it."""
        out = []
        for t in rows[i][1]["terms"]:
            if t["kind"] == C:
                j = standing.get(t["source"])
                out.append(j if j is not None and (rows[i][0] == "user" or rows[j][0] == "model")
                           else None)
        return out

    def reach(i: int) -> set[int]:
        seen: set[int] = set()
        todo = [j for j in targets(i) if j is not None]
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                todo.extend(k for k in targets(j) if k is not None)
        return seen

    on_cycle = {i for i in standing.values() if i in reach(i)}
    for i in standing.values():
        if i in on_cycle or reach(i) & on_cycle:
            verdict[i] = "cycle"
    changed = True
    while changed:
        changed = False
        for i in standing.values():
            if i not in verdict and any(j is None or j in verdict for j in targets(i)):
                verdict[i] = "unresolved"
                changed = True
    out: dict[str, dict[str, str]] = {"user": {}, "model": {}}
    for i, reason in sorted(verdict.items()):
        out[rows[i][0]][rows[i][1]["name"]] = reason
    return out


class Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def cells(field: dict[str, Any], scale: float = 1.0) -> Cells:
    out: Cells = {}
    for key, row in zip(field["keys"], field["values"]):
        for label, value in zip(field["components"], row):
            out[(tuple(key), label)] = NAN if value is None else scale * value
    return out


def union(maps: list[Cells]) -> list[tuple[tuple[int, ...], str]]:
    keys = sorted({k for m in maps for (k, _) in m})
    labels: list[str] = []
    for m in maps:
        for (_, label) in m:
            if label not in labels:
                labels.append(label)
    return [(k, label) for k in keys for label in labels]


def pick(m: Cells, at: tuple[tuple[int, ...], str]) -> float:
    return m.get(at, NAN)


def nanpass(values: list[float], f) -> float:
    return NAN if any(math.isnan(v) for v in values) else f(values)


def rho(wi: float, wj: float, xi: float, xj: float) -> float:
    if wi == wj and xi == xj:
        return 1.0
    r = wj / wi
    num = 8.0 * math.sqrt(xi * xj) * (xi + r * xj) * r ** 1.5
    den = (1 - r * r) ** 2 + 4 * xi * xj * r * (1 + r * r) + 4 * (xi * xi + xj * xj) * r * r
    return num / den


def cqc_value(values: list[float], omega: list[float], xi: list[float]) -> float:
    def f(v: list[float]) -> float:
        total = 0.0
        for i in range(len(v)):
            for j in range(len(v)):
                total += v[i] * (1.0 if i == j else rho(omega[i], omega[j], xi[i], xi[j])) * v[j]
        return math.sqrt(max(0.0, total))
    return nanpass(values, f)


class Reference:
    def __init__(self, quantity: str) -> None:
        q = QUANTITIES[quantity]
        self.quantity = quantity
        self.linear = q["linear"]
        self.scale = q["scale"]
        self.stored = q["stored"]
        verdicts = table_verdicts()
        self.refused = set(verdicts["user"]) | set(verdicts["model"])
        self.rows = {c["name"]: c for c in [*MODEL_COMBINATIONS, *COMBINATIONS]
                     if c["name"] not in self.refused}
        self.memo: dict[str, Cells] = {}
        self.key_names = ["id"]

    def state(self, addr: str) -> Cells:
        field = STATES.get(addr)
        if field is None and self.quantity == "recorded":
            field = OTHER_STATES.get(addr)
        if field is None:
            raise Refused("unresolved")
        self.key_names = field["key_names"]
        return cells(field, self.scale)

    def term(self, t: dict[str, Any]) -> Cells:
        if t["factor"] != 1.0 and not self.linear:
            raise Refused("not-linear")  # a factor scales a quantity that is not linear
        if t["kind"] == C:
            m = self.value(t["source"])
        else:
            m = self.state(address(t["kind"], t["source"], t["step"], t["mode"]))
        return {k: t["factor"] * v for k, v in m.items()}

    def check_a5(self, name: str) -> None:
        """A5 over every combination the value needs, before any value is read."""
        todo, seen = [name], set()
        while todo:
            current = todo.pop()
            if current in seen or current in self.stored or current not in self.rows:
                continue
            seen.add(current)
            row = self.rows[current]
            if not self.linear and (row["rule"] != "envelope"
                                    or any(t["factor"] != 1.0 for t in row["terms"])):
                raise Refused("not-linear")
            todo.extend(t["source"] for t in row["terms"] if t["kind"] == C)

    def value(self, name: str) -> Cells:
        if name in self.memo:
            return self.memo[name]
        if name in self.refused:
            raise Refused("refused")
        if name not in self.rows:
            raise Refused("unresolved")
        row = self.rows[name]
        if name in self.stored:
            out = cells(self.stored[name])
        else:
            if row["rule"] != "envelope" and not self.linear:
                raise Refused("not-linear")
            out = self.apply(row)
        self.memo[name] = out
        return out

    def whole_case(self, row: dict[str, Any], t: dict[str, Any]) -> Cells:
        if self.quantity != "recorded":
            # A user result has no case table of its own: /spectra holds it (A5).
            raise Refused("unresolved")
        c = CASES[t["source"]]
        maps = [cells(f, self.scale) for f in c["increments"]]
        # Group modes that share sa_period > 0, in the first one's slot.
        slots: list[list[int]] = []
        first_of: dict[float, int] = {}
        for i, period in enumerate(c["sa_period"]):
            if period > 0 and period in first_of:
                slots[first_of[period]].append(i)
            else:
                if period > 0:
                    first_of[period] = len(slots)
                slots.append([i])
        lead = [s[0] for s in slots]
        omega, xi = [], []
        for i in lead:
            lam = c["lambda"][i]
            if not lam > 0:
                raise Refused("bad-param")
            period = c["sa_period"][i]
            omega.append(2 * math.pi / period if period > 0 else math.sqrt(lam))
            xi.append(row["damping"] if row["damping"] is not None else c["damping"][i])
        out: Cells = {}
        for at in union(maps):
            v = [sum(pick(maps[i], at) for i in s) for s in slots]
            if row["rule"] == "srss":
                r = nanpass(v, lambda v: math.sqrt(sum(x * x for x in v)))
            elif row["rule"] == "abs_sum":
                r = nanpass(v, lambda v: sum(abs(x) for x in v))
            else:
                r = cqc_value(v, omega, xi)
            out[at] = abs(t["factor"]) * r
        return out

    def apply(self, row: dict[str, Any]) -> Cells:
        rule = row["rule"]
        whole = [t for t in row["terms"] if t["kind"] == "spectrum" and t["mode"] == 0]
        if whole:
            return self.whole_case(row, whole[0])
        maps = [self.term(t) for t in row["terms"]]
        omega: list[float] = []
        xi: list[float] = []
        if rule == "cqc":
            for t in row["terms"]:
                if t["kind"] == M:
                    lam = EIGENVALUES[address(M, t["source"], t["step"], t["mode"])]
                    damp = row["damping"]
                else:
                    c = CASES[t["source"]]
                    i = c["mode"].index(t["mode"])
                    lam = c["lambda"][i]
                    damp = row["damping"] if row["damping"] is not None else c["damping"][i]
                omega.append(math.sqrt(lam))
                xi.append(damp)
        out: Cells = {}
        p = row["fraction"]
        for at in union(maps):
            v = [pick(m, at) for m in maps]
            if rule == "linear":
                r = nanpass(v, sum)
            elif rule == "srss":
                r = nanpass(v, lambda v: math.sqrt(sum(x * x for x in v)))
            elif rule == "abs_sum":
                r = nanpass(v, lambda v: sum(abs(x) for x in v))
            elif rule == "cqc":
                r = cqc_value(v, omega, xi)
            elif rule == "directional":
                r = nanpass(v, lambda v: max(
                    abs(v[a]) + p * sum(abs(v[b]) for b in range(len(v)) if b != a)
                    for a in range(len(v))))
            elif row["envelope"] == "max":
                r = nanpass(v, max)
            elif row["envelope"] == "min":
                r = nanpass(v, min)
            else:
                r = nanpass(v, lambda v: max(abs(x) for x in v))
            out[at] = r
        return out


def signed(rows: dict[str, dict[str, Any]], name: str) -> bool:
    row = rows[name]
    return row["rule"] == "linear" and all(
        signed(rows, t["source"]) if t["kind"] == C else True for t in row["terms"])


def written(m: Cells, key_names: list[str], is_signed: bool, badges: list[str]) -> dict[str, Any]:
    at = union([m])
    keys = sorted({k for k, _ in at})
    labels = [label for k, label in at if k == keys[0]]
    return {
        "signed": is_signed,
        "badges": badges,
        "key_names": key_names,
        "keys": [list(k) for k in keys],
        "components": labels,
        "values": [[None if math.isnan(m[(k, label)]) else m[(k, label)] for label in labels]
                   for k in keys],
    }


def golden(quantity: str, name: str) -> dict[str, Any]:
    ref = Reference(quantity)
    try:
        if name in ref.rows:
            ref.check_a5(name)
        m = ref.value(name)
    except Refused as exc:
        return {"refused": exc.reason}
    row = ref.rows[name]
    badge = row["rule"] == "linear" and any(t["kind"] == S for t in row["terms"])
    return written(m, ref.key_names, signed(ref.rows, name),
                   ["superposition assumes linearity"] if badge else [])


def golden_lone(quantity: str, addr: str) -> dict[str, Any]:
    kind, _, _, mode = addr.split("|")
    if kind == "spectrum" and mode == "0":
        return {"refused": "bad-term"}  # a whole case is many states
    ref = Reference(quantity)
    try:
        m = ref.state(addr)
    except Refused as exc:
        return {"refused": exc.reason}
    return written(m, ref.key_names, True, [])


def build() -> dict[str, Any]:
    goldens: dict[str, dict[str, Any]] = {}
    for quantity in QUANTITIES:
        names = [c["name"] for c in COMBINATIONS] if quantity == "recorded" else USER_COMBINATIONS
        goldens[quantity] = {name: golden(quantity, name) for name in names}
    lone = {q: {a: golden_lone(q, a) for a in addrs} for q, addrs in LONE.items()}
    return {
        "version": 1,
        "rtol": 1e-12,
        "about": "ADR 0130 D4 conformance fixture; written by make_conformance.py. "
                 "null is NaN.",
        "states": STATES,
        "other_states": OTHER_STATES,
        "eigenvalues": EIGENVALUES,
        "cases": CASES,
        "combinations": COMBINATIONS,
        "model_combinations": MODEL_COMBINATIONS,
        "quantities": QUANTITIES,
        "table": table_verdicts(),
        "goldens": goldens,
        "lone": lone,
    }


def dumps(doc: dict[str, Any]) -> str:
    def clean(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(doc), indent=1, sort_keys=False, allow_nan=False) + "\n"


if __name__ == "__main__":
    OUT.write_text(dumps(build()), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
