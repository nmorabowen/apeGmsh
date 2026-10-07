"""
TagAllocator — sequential, per-kind, 1-based tag allocation.

The bridge owns this. Primitives never see it directly; the bridge
calls :meth:`allocate` (or :meth:`allocate_for`) at registration time
and stashes the result.

OpenSees tags are scoped per command kind: a uniaxialMaterial with
tag 1, a section with tag 1, and an element with tag 1 do not
collide. The allocator therefore keeps an independent counter per
kind string.

Idempotency: :meth:`allocate_for` returns the same tag if the same
primitive instance is registered twice (lookup keyed on ``id()``).
This protects against accidental double-registration of a primitive
the user constructed standalone (P11) and then passed through the
namespace API.

The tag law (ADR 0114 D4, amended): a tag is written once, by the
bridge's build. The build-time tag plan (``_internal/tag_plan.py``)
mints with one of these allocators and then calls :meth:`freeze`; every
minting method of a frozen allocator raises :class:`TagLawError`, so a
mint after the plan raises where it happens. :meth:`fork` copies the
state with chosen kinds frozen: the safety net while the emit paths
move their minting into the plan one family at a time.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import NoReturn


class TagLawError(RuntimeError):
    """A tag was minted after the tag plan froze its allocator (or kind)."""


class TagAllocator:
    """Per-kind sequential 1-based tag allocator."""

    __slots__ = (
        "_counters", "_assignments", "_frozen", "_frozen_kinds", "_forked",
    )

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        # id(primitive) -> assigned tag. The kind context is implicit:
        # a primitive is tagged exactly once, in exactly one kind.
        self._assignments: dict[int, int] = {}
        # Whole-allocator freeze (:meth:`freeze`) and the kinds a
        # :meth:`fork` froze. Either makes a mint raise TagLawError.
        self._frozen: bool = False
        self._frozen_kinds: frozenset[str] = frozenset()
        # Set by :meth:`fork`: a fork refuses :meth:`reset`.
        self._forked: bool = False

    # ------------------------------------------------------------------
    # The freeze
    # ------------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        """``True`` once :meth:`freeze` has been called."""
        return self._frozen

    @property
    def frozen_kinds(self) -> frozenset[str]:
        """The kinds frozen by :meth:`fork` (empty for a fresh allocator)."""
        return self._frozen_kinds

    def freeze(self) -> None:
        """Refuse every later mint, in every kind.

        After this, :meth:`allocate`, :meth:`allocate_block`,
        :meth:`allocate_for`, :meth:`reserve_through` and :meth:`reset`
        raise :class:`TagLawError`. Reads (:meth:`last`, :meth:`tag_for`)
        and :meth:`fork` stay legal. Idempotent.
        """
        self._frozen = True

    def fork(self, frozen_kinds: Iterable[str] = ()) -> TagAllocator:
        """A mutable copy of this allocator with ``frozen_kinds`` frozen.

        The copy continues every counter and assignment from here. A
        mint in a kind of ``frozen_kinds`` (or in a kind this allocator's
        own fork froze) raises :class:`TagLawError`; every other kind
        mints on. The copy is not whole-frozen even when this allocator
        is, and this allocator is never changed by the copy's mints. A
        copy refuses :meth:`reset`, frozen kinds or not: clearing it would
        re-mint tags its parent already handed out.
        """
        kinds = frozenset(frozen_kinds)
        for k in kinds:
            if not isinstance(k, str) or not k:
                raise TypeError(
                    f"fork: a frozen kind must be a non-empty str, got {k!r}."
                )
        child = TagAllocator()
        child._counters = dict(self._counters)
        child._assignments = dict(self._assignments)
        child._frozen_kinds = self._frozen_kinds | kinds
        child._forked = True
        return child

    def _refuse(self, kind: str | None, verb: str) -> NoReturn:
        """Raise the :class:`TagLawError` for a refused mint."""
        if self._frozen:
            raise TagLawError(
                f"{verb}({kind!r}) after the tag plan froze this allocator: "
                "a tag is minted once, by the build's tag plan (ADR 0114 "
                "D4). Move this mint into plan_tags."
            )
        if kind is not None and kind in self._frozen_kinds:
            raise TagLawError(
                f"{verb}({kind!r}): kind {kind!r} is planned and frozen in "
                "this allocator, so the emit path must read its tag from "
                "the tag plan instead of minting one (ADR 0114 D4)."
            )
        if kind is None and (self._frozen_kinds or self._forked):
            raise TagLawError(
                f"{verb}() on a forked allocator (frozen kinds "
                f"{sorted(self._frozen_kinds)}): clearing it would let it "
                "re-mint tags the plan already handed out (ADR 0114 D4)."
            )
        raise AssertionError("_refuse called for an allowed mint")

    # ------------------------------------------------------------------
    # Minting
    # ------------------------------------------------------------------

    def allocate(self, kind: str) -> int:
        """Return the next 1-based tag for ``kind`` and bump the counter."""
        if self._frozen or kind in self._frozen_kinds:
            self._refuse(kind, "allocate")
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        return n

    def allocate_block(self, kind: str, n: int) -> int:
        """Reserve ``n`` contiguous tags for ``kind``; return the first.

        Equivalent to calling :meth:`allocate` ``n`` times and keeping
        the first result — the counter advances by exactly ``n`` and the
        reserved tags are ``[first, first + n)``. Used by
        :func:`allocate_element_tags` (ADR 0065 v2 /
        plan_emit_memory_columnar.md B1) to reserve a whole spec's
        element tags in one call rather than boxing one Python ``int``
        per element; the columnar plan then derives row ``i``'s tag as
        ``tag_start + i`` positionally. Counter semantics are identical
        to the per-call loop (nothing observes per-call side effects —
        ``allocate`` only bumps ``_counters[kind]``), so byte-identical
        tag numbering is preserved.

        ``n == 0`` reserves nothing and returns the next tag that *would*
        be allocated (the counter is untouched).
        """
        if self._frozen or kind in self._frozen_kinds:
            self._refuse(kind, "allocate_block")
        if n < 0:
            raise ValueError(f"allocate_block: n must be >= 0, got {n}.")
        base = self._counters.get(kind, 0)
        if n == 0:
            return base + 1
        self._counters[kind] = base + n
        return base + 1

    def last(self, kind: str) -> int:
        """The last tag handed out for ``kind`` (``0`` before the first)."""
        return self._counters.get(kind, 0)

    def reserve_through(self, kind: str, last: int) -> None:
        """Mark every tag ``<= last`` of ``kind`` as taken.

        Raises the counter to ``max(counter, last)`` so the next
        :meth:`allocate` / :meth:`allocate_block` returns a tag above
        ``last``; never lowers it. Used by ``element_tags="fem"`` (ADR
        0111 D2) to put every bridge-synthesised element above the
        model's FEM element ids, which the plan uses verbatim as tags.
        """
        if self._frozen or kind in self._frozen_kinds:
            self._refuse(kind, "reserve_through")
        if last > self._counters.get(kind, 0):
            self._counters[kind] = int(last)

    def allocate_for(self, primitive: object, kind: str) -> int:
        """Allocate a tag for a primitive, idempotent on repeat calls.

        If ``primitive`` has already been allocated, return the same
        tag and do not bump the counter. This keeps standalone-then-
        registered primitives (P11) safe against accidental
        double-registration.

        A frozen allocator (or kind) raises :class:`TagLawError` even
        when ``primitive`` already has a tag: read it with
        :meth:`tag_for` instead.
        """
        if self._frozen or kind in self._frozen_kinds:
            self._refuse(kind, "allocate_for")
        prev = self._assignments.get(id(primitive))
        if prev is not None:
            return prev
        tag = self.allocate(kind)
        self._assignments[id(primitive)] = tag
        return tag

    def tag_for(self, primitive: object) -> int | None:
        """Return the previously allocated tag for ``primitive``, or
        ``None`` if it has not been allocated yet."""
        return self._assignments.get(id(primitive))

    def reset(self) -> None:
        """Clear all counters and assignments — fresh allocator state.

        Refused on a frozen allocator and on every fork, frozen kinds or
        not: clearing it would let the next mint hand out a planned tag a
        second time.
        """
        if self._frozen or self._forked:
            self._refuse(None, "reset")
        self._counters.clear()
        self._assignments.clear()
