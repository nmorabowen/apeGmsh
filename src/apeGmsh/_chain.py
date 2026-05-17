"""Shared fluent-selection idiom — the ``SelectionChain`` mixin.

This is a **leaf module**: it imports only the standard library.  It
must never import ``apeGmsh.core``/``mesh``/``viz``/``results`` (that
would invert the load-bearing eager/deferred import polarity — see
``docs/plans/selection-unification.md`` §3, and the
``tests/test_import_dag_polarity.py`` guard).

``SelectionChain`` owns *only* what is genuinely identical across the
four selection domains (geometry / mesh / broker / results):

* the daisy-chain protocol — every refining verb returns a new
  instance of the *same* concrete subclass (covariant), so calls
  compose: ``sel.in_box(...).on_plane(...).nearest_to(...)``;
* set algebra — ``| & - ^`` plus the named aliases, with one fixed
  dedup law (insertion-order preserving);
* the public verb *names*, enforced at class-definition time by
  :meth:`__init_subclass__` (a subclass that drops or renames a verb,
  or forgets ``FAMILY``/a hook, fails at import — stronger than a CI
  test).

The behaviour of the spatial verbs differs per ``FAMILY`` and is
delegated to abstract hooks.  ``FAMILY == "entity"`` (geometry) tests
CAD-entity bounding boxes via Gmsh and cannot honour a half-open /
``inclusive=`` knob; ``FAMILY == "point"`` (mesh / broker / results)
tests node coordinates or element centroids.  The per-family contract
is asserted by the CI contract test, never by cross-family identity.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Iterator

__all__ = ["SelectionChain", "VALID_FAMILIES", "REQUIRED_VERBS"]

VALID_FAMILIES = ("entity", "point")

#: Public verb surface every concrete chain must expose.  Locked here;
#: ``__init_subclass__`` makes a missing/renamed verb an ImportError.
REQUIRED_VERBS = (
    "in_box", "in_sphere", "on_plane", "nearest_to", "where",
    "union", "intersect", "difference",
)

#: Hooks a concrete subclass must implement (not inherit as the base
#: ``NotImplementedError`` stub).
_REQUIRED_HOOKS = (
    "_coords_of", "_spatial_box", "_spatial_sphere", "_spatial_plane",
    "_materialize",
)


class SelectionChain:
    """Mixin: chaining + set-algebra + name-enforcement.

    A chain carries an *ordered, de-duplicated* tuple of opaque atoms
    (node ids, element ids, or ``(dim, tag)`` dimtags — all hashable)
    and an opaque ``_engine`` back-reference the subclass uses to fetch
    coordinates and to materialise a terminal value.
    """

    FAMILY: ClassVar[str] = ""  # concrete subclasses set "entity"|"point"

    __slots__ = ("_items", "_engine")

    # ── construction ────────────────────────────────────────
    def __init__(self, atoms: Iterable[Any] = (), *, _engine: Any = None) -> None:
        self._items: tuple = self._dedupe(atoms)
        self._engine = _engine

    @staticmethod
    def _dedupe(atoms: Iterable[Any]) -> tuple:
        """Insertion-order-preserving de-duplication (the one law)."""
        return tuple(dict.fromkeys(atoms))

    def _wrap(self, atoms: Iterable[Any]) -> "SelectionChain":
        """New chain of *this exact* subclass, same engine (covariant)."""
        return type(self)(atoms, _engine=self._engine)

    def _atoms(self) -> tuple:
        return self._items

    # ── definition-time name enforcement (ratified R2) ──────
    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # Abstract intermediates (no FAMILY) are exempt; only concrete
        # leaves are checked.
        fam = cls.__dict__.get("FAMILY", cls.FAMILY)
        if not fam:
            return
        if fam not in VALID_FAMILIES:
            raise TypeError(
                f"{cls.__name__}.FAMILY={fam!r} invalid; "
                f"expected one of {VALID_FAMILIES}."
            )
        for verb in REQUIRED_VERBS:
            if not callable(getattr(cls, verb, None)):
                raise TypeError(
                    f"{cls.__name__} is missing required selection verb "
                    f"{verb!r}. The unified chain surface is fixed: "
                    f"{REQUIRED_VERBS}."
                )
        for hook in _REQUIRED_HOOKS:
            impl = getattr(cls, hook, None)
            if impl is None or impl is getattr(SelectionChain, hook, None):
                raise TypeError(
                    f"{cls.__name__} must implement the {hook!r} hook "
                    f"(FAMILY={fam!r})."
                )

    # ── abstract per-domain hooks ───────────────────────────
    def _coords_of(self, atoms: tuple):
        raise NotImplementedError

    def _spatial_box(self, atoms: tuple, lo, hi, *, inclusive: bool) -> tuple:
        raise NotImplementedError

    def _spatial_sphere(self, atoms: tuple, center, radius: float) -> tuple:
        raise NotImplementedError

    def _spatial_plane(self, atoms: tuple, point, normal, tol: float) -> tuple:
        raise NotImplementedError

    def _materialize(self):
        raise NotImplementedError

    # ── public verb surface (identical names everywhere) ────
    def in_box(self, lo, hi, *, inclusive: bool = False) -> "SelectionChain":
        return self._wrap(
            self._spatial_box(self._items, lo, hi, inclusive=inclusive)
        )

    def in_sphere(self, center, radius: float) -> "SelectionChain":
        return self._wrap(self._spatial_sphere(self._items, center, radius))

    def on_plane(self, point, normal, *, tol: float) -> "SelectionChain":
        return self._wrap(self._spatial_plane(self._items, point, normal, tol))

    def nearest_to(self, point, *, count: int = 1) -> "SelectionChain":
        return self._wrap(self._nearest(self._items, point, count))

    def where(self, predicate) -> "SelectionChain":
        """Keep atoms whose coordinate row satisfies ``predicate``."""
        coords = self._coords_of(self._items)
        keep = [a for a, xyz in zip(self._items, coords) if predicate(xyz)]
        return self._wrap(keep)

    # nearest is point-family default; entity family overrides _nearest
    def _nearest(self, atoms: tuple, point, count: int) -> tuple:
        import math

        coords = self._coords_of(atoms)
        order = sorted(
            range(len(atoms)),
            key=lambda i: (
                math.fsum((coords[i][k] - point[k]) ** 2 for k in range(3)),
                i,  # deterministic tie-break: lowest index
            ),
        )
        return tuple(atoms[i] for i in order[:count])

    # ── set algebra (one dedup law; cross-type is loud) ─────
    def _compatible(self, other: "SelectionChain") -> None:
        if type(self) is not type(other):
            raise TypeError(
                f"cannot combine {type(self).__name__} with "
                f"{type(other).__name__}: selection set-algebra requires "
                f"the same chain type (same level + atom space)."
            )
        if self._engine is not other._engine:
            raise TypeError(
                "cannot combine chains bound to different engines "
                "(different FEMData / model / results)."
            )

    def union(self, other: "SelectionChain") -> "SelectionChain":
        self._compatible(other)
        return self._wrap(self._items + other._items)

    def intersect(self, other: "SelectionChain") -> "SelectionChain":
        self._compatible(other)
        keep = set(other._items)
        return self._wrap(a for a in self._items if a in keep)

    def difference(self, other: "SelectionChain") -> "SelectionChain":
        self._compatible(other)
        drop = set(other._items)
        return self._wrap(a for a in self._items if a not in drop)

    def symmetric_difference(self, other: "SelectionChain") -> "SelectionChain":
        self._compatible(other)
        a, b = set(self._items), set(other._items)
        only = a ^ b
        # preserve insertion order across self then other
        return self._wrap(
            x for x in (self._items + other._items) if x in only
        )

    __or__ = union
    __and__ = intersect
    __sub__ = difference
    __xor__ = symmetric_difference

    # ── terminal / introspection ────────────────────────────
    def result(self):
        """Materialise the domain-specific terminal value."""
        return self._materialize()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} FAMILY={self.FAMILY!r} "
            f"n={len(self._items)}>"
        )
