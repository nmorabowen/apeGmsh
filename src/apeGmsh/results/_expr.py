"""User-defined scalar expressions — computed on read from named columns.

Sibling of :mod:`._derived`: same "combine named ``(T, N)`` arrays into a
``(T, N)`` scalar at read time" contract, but the recipe is a user-supplied
string instead of a hard-coded invariant. See ADR 0076.

The expression is parsed to a **restricted AST** (never :func:`eval`) and
only a fixed allow-list of node types and functions is admitted — every
one array-safe and *elementwise*, so a compiled expression always
preserves the operands' shape. In particular:

* ``and`` / ``or`` (``BoolOp``) and ``a if c else b`` (``IfExp``) are
  **rejected**, not lowered — they dispatch on Python truthiness, which
  raises on a numpy array. Use the ``where(cond, a, b)`` function and
  bitwise ``&`` / ``|`` on comparison results instead.
* ``min`` / ``max`` are **not** exposed: numpy's are reductions (they
  would collapse the field) and Python's are variadic. The elementwise
  two-argument ``minimum`` / ``maximum`` are exposed instead.
* ``mag(<vector>)`` is a special form: it takes a bare nodal vector-family
  name (``velocity``, ``displacement``, ``force``, …) and expands to the
  Euclidean norm of the family's recorded components — ``sqrt(vx**2 +
  vy**2 + vz**2)`` — clipping to the present axes (ADR 0076 Slice 4).

Public surface::

    compile_expr(name, expr, *, available, label, units) -> ExprDef
    evaluate(exprdef, namespace)                          -> ndarray
    ExprError
    ExprDef
"""
from __future__ import annotations

import ast
import dataclasses
from typing import Callable, Iterable

import numpy as np
from numpy import ndarray

from .. import _vocabulary

__all__ = ["ExprError", "ExprDef", "compile_expr", "evaluate"]


class ExprError(ValueError):
    """A user expression is malformed, references an unknown operand, or
    its operands do not align at evaluation time."""


# Elementwise, shape-preserving numpy functions: call name → (impl, arity).
# NO reductions (np.min/max/sum), NO variadic builtins — every entry maps a
# fixed arity of array operands to an array of the same shape.
_FUNCS: dict[str, tuple[Callable, int]] = {
    "sqrt": (np.sqrt, 1),
    "abs": (np.abs, 1),
    "exp": (np.exp, 1),
    "log": (np.log, 1),
    "sin": (np.sin, 1),
    "cos": (np.cos, 1),
    "tan": (np.tan, 1),
    "sign": (np.sign, 1),
    "hypot": (np.hypot, 2),
    "minimum": (np.minimum, 2),   # elementwise pair — NOT the np.min reduction
    "maximum": (np.maximum, 2),
    "clip": (np.clip, 3),         # clip(x, lo, hi)
    "where": (np.where, 3),       # where(cond, a, b) — the elementwise if/else
}

_BINOPS: dict[type, Callable] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.FloorDiv: np.floor_divide,
    ast.Mod: np.mod,
    ast.Pow: np.power,
    ast.BitAnd: np.bitwise_and,   # elementwise combine of comparison results
    ast.BitOr: np.bitwise_or,
}

_UNARYOPS: dict[type, Callable] = {
    ast.UAdd: np.positive,
    ast.USub: np.negative,
}

_CMPOPS: dict[type, Callable] = {
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


@dataclasses.dataclass(frozen=True)
class ExprDef:
    """A compiled, validated user scalar expression.

    ``operands`` is the set of identifiers the expression actually
    references (a subset of the ``available`` names it was compiled
    against). ``label`` / ``units`` are display-only metadata for the
    viewer picker / legend and never affect evaluation. ``_fn`` is the
    compiled evaluator closure; it is excluded from equality and is not
    serialized (the viewer rebuilds it from ``expr``).
    """

    name: str
    expr: str
    operands: frozenset[str]
    label: str
    units: str
    _fn: Callable[[dict[str, ndarray]], ndarray] = dataclasses.field(
        repr=False, compare=False,
    )


def compile_expr(
    name: str, expr: str, *, available: Iterable[str],
    label: str | None = None, units: str | None = None,
) -> ExprDef:
    """Parse and validate ``expr`` into an :class:`ExprDef`.

    ``available`` is the set of operand names the expression may
    reference (the composite's stored + derived + already-registered
    custom components). Raises :class:`ExprError` on a parse failure, a
    disallowed syntax node, an unknown function, or an identifier not in
    ``available``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(
            f"could not parse expression for {name!r}: {exc.msg}"
        ) from exc

    avail = set(available)
    operands: set[str] = set()
    fn = _compile(tree.body, available=avail, operands=operands, ctx=name)
    return ExprDef(
        name=name,
        expr=expr,
        operands=frozenset(operands),
        label=label if label is not None else name,
        units=units if units is not None else "",
        _fn=fn,
    )


def evaluate(exprdef: ExprDef, namespace: dict[str, ndarray]) -> ndarray:
    """Evaluate ``exprdef`` against ``namespace`` (operand name → array).

    Every name in ``exprdef.operands`` must be present in ``namespace``.
    Enforces operand-shape agreement first (so a coverage mismatch raises
    a legible :class:`ExprError`, not a raw numpy broadcast error), then
    returns a float64 array of the operands' shape.
    """
    _assert_aligned(exprdef, namespace)
    result = exprdef._fn(namespace)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _assert_aligned(exprdef: ExprDef, namespace: dict[str, ndarray]) -> None:
    """Raise unless every referenced operand array shares one shape.

    Unlike the derived scalars — which only ever combine co-recorded
    components of one tensor — a custom expression can name two
    independently recorded fields whose selections cover different point
    counts. Catch that here with a readable error instead of letting
    numpy broadcast (or reject) the mismatch downstream.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for nm in exprdef.operands:
        try:
            shapes[nm] = np.asarray(namespace[nm]).shape
        except KeyError:  # pragma: no cover — composite guarantees presence
            raise ExprError(
                f"operand {nm!r} of {exprdef.name!r} was not supplied."
            ) from None
    if len(set(shapes.values())) > 1:
        desc = ", ".join(f"{nm!r} {shp}" for nm, shp in sorted(shapes.items()))
        raise ExprError(
            f"operands of {exprdef.name!r} cover different points for this "
            f"selection: {desc}. A custom expression can only combine fields "
            f"recorded on the same points — narrow the selection so every "
            f"operand is present, or split the expression."
        )


def _compile(
    node: ast.AST, *, available: set[str], operands: set[str], ctx: str,
) -> Callable[[dict[str, ndarray]], ndarray]:
    """Recursively lower an allow-listed AST node to an evaluator closure.

    Any node type outside the allow-list — attribute access, subscript,
    comprehensions, lambdas, ``BoolOp`` / ``IfExp``, f-strings, starred
    args — falls through to the final ``ExprError``.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ExprError(
                f"only numeric literals are allowed in {ctx!r} "
                f"(got {node.value!r})."
            )
        value = float(node.value)
        return lambda ns: value

    if isinstance(node, ast.Name):
        nm = node.id
        if nm not in available:
            raise ExprError(
                f"unknown operand {nm!r} in {ctx!r}. Available components: "
                f"{sorted(available)}."
            )
        operands.add(nm)
        return lambda ns, _nm=nm: ns[_nm]

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExprError(
                f"operator {type(node.op).__name__} is not allowed in {ctx!r}."
            )
        lhs = _compile(node.left, available=available, operands=operands, ctx=ctx)
        rhs = _compile(node.right, available=available, operands=operands, ctx=ctx)
        return lambda ns: op(lhs(ns), rhs(ns))

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:  # e.g. ast.Not, ast.Invert
            raise ExprError(
                f"unary {type(node.op).__name__} is not allowed in {ctx!r}."
            )
        operand = _compile(
            node.operand, available=available, operands=operands, ctx=ctx,
        )
        return lambda ns: op(operand(ns))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExprError(
                f"chained comparisons are not supported in {ctx!r}; "
                f"split into two comparisons combined with & or |."
            )
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise ExprError(
                f"comparison {type(node.ops[0]).__name__} is not allowed "
                f"in {ctx!r}."
            )
        lhs = _compile(node.left, available=available, operands=operands, ctx=ctx)
        rhs = _compile(
            node.comparators[0], available=available, operands=operands, ctx=ctx,
        )
        return lambda ns: op(lhs(ns), rhs(ns))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExprError(f"only direct function calls are allowed in {ctx!r}.")
        fname = node.func.id
        if fname == "mag":
            return _compile_mag(
                node, available=available, operands=operands, ctx=ctx,
            )
        entry = _FUNCS.get(fname)
        if entry is None:
            raise ExprError(
                f"unknown function {fname!r} in {ctx!r}. Allowed: "
                f"{sorted(_FUNCS)}."
            )
        if node.keywords:
            raise ExprError(
                f"function {fname!r} does not take keyword arguments in {ctx!r}."
            )
        impl, arity = entry
        if len(node.args) != arity:
            raise ExprError(
                f"function {fname!r} takes {arity} argument(s), "
                f"got {len(node.args)} in {ctx!r}."
            )
        args = [
            _compile(a, available=available, operands=operands, ctx=ctx)
            for a in node.args
        ]
        return lambda ns: impl(*(a(ns) for a in args))

    raise ExprError(
        f"disallowed syntax {type(node).__name__} in {ctx!r}. Expressions "
        f"may use + - * / // % **, comparisons, & |, numeric literals, the "
        f"operand names, mag(<vector>), and {sorted(_FUNCS)}. Note: "
        f"`and`/`or`/`if-else` are not allowed — use where(cond, a, b) "
        f"and & / |."
    )


def _compile_mag(
    node: ast.Call, *, available: set[str], operands: set[str], ctx: str,
) -> Callable[[dict[str, ndarray]], ndarray]:
    """Lower ``mag(<vector>)`` to the Euclidean norm of a vector family.

    ``mag(velocity)`` expands to ``sqrt(velocity_x**2 + velocity_y**2 +
    velocity_z**2)`` over whichever of the family's components are
    actually recorded (so a 2-D model with only ``_x`` / ``_y`` yields
    the in-plane norm). The single argument must be a bare vector-family
    name (``displacement``, ``velocity``, ``force``, ``rotation``,
    ``moment``, …) — not an expression, and not the combined
    ``reaction`` shorthand.
    """
    if node.keywords or len(node.args) != 1 or not isinstance(
        node.args[0], ast.Name
    ):
        raise ExprError(
            f"mag(...) takes a single vector-family name in {ctx!r}, "
            f"e.g. mag(velocity)."
        )
    family = node.args[0].id
    components = _vocabulary.vector_family(family)
    if components is None:
        raise ExprError(
            f"mag({family}) in {ctx!r}: {family!r} is not a vector family. "
            f"Use one of the nodal vector families (displacement, velocity, "
            f"acceleration, force, rotation, moment, …)."
        )
    present = [c for c in components if c in available]
    if not present:
        raise ExprError(
            f"mag({family}) in {ctx!r}: none of {list(components)} are "
            f"available components (available: {sorted(available)})."
        )
    operands.update(present)

    def _mag(ns: dict[str, ndarray], _present=tuple(present)) -> ndarray:
        acc = None
        for c in _present:
            v = ns[c]
            acc = v * v if acc is None else acc + v * v
        return np.sqrt(acc)

    return _mag
