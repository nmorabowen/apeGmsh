"""Declaration stores — the one path by which a composite records intent.

Eight session composites record intent that the next
``g.mesh.queries.get_fem_data()`` resolves into the FEMData broker:
``g.constraints``, ``g.reinforce``, ``g.embed``, ``g.rebar``,
``g.loads``, ``g.displacements``, ``g.masses`` and
``g.decoupled_nodes``.  The session caches that extraction (ADR 0038)
and hands back the same snapshot until ``_bump_fem_counter()`` marks it
stale, so every declaration has to bump.  Otherwise a def declared after
the first extraction is stored but left out of every later snapshot.

The bump used to be a line written beside each ``append``.  Five verbs
added after the cache (``g.constraints.contact`` / ``contact_plane`` /
``interface``, ``g.reinforce`` and ``g.embed``) never got one, and no
``clear()`` had one.  :meth:`_DeclarationsMixin._declare` makes storing
and invalidating one step, and ``tests/test_declaration_coverage.py``
holds every public verb to it.
"""
from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, TypeVar

from apeGmsh._internal.provenance import capture as _capture_provenance

from ._compose_errors import ChainPhaseError, is_kernelless_session

_D = TypeVar("_D")

#: Store attribute -> the provenance family its defs are recorded under
#: (declaration path ``neutral/<family>/<name|#k>``, ADR 0112 D3).  A store
#: missing here makes :meth:`_DeclarationsMixin._declare` raise, so a new
#: store cannot silently go unrecorded.
_PROVENANCE_FAMILY: dict[str, str] = {
    "constraint_defs": "constraints",
    "_bc_defs": "bcs",
    "contact_defs": "contacts",
    "contact_plane_defs": "contact_planes",
    "interface_defs": "interfaces",
    "node_defs": "decoupled_nodes",
    "disp_defs": "displacements",
    "embed_defs": "embeds",
    "load_defs": "loads",
    "mass_defs": "masses",
    "placements": "rebar",
    "_emit_members": "rebar_members",
    "reinforce_defs": "reinforcements",
}


def _declaration_name(defn: object) -> str | None:
    """The user's name for ``defn``: its ``name`` field, else ``label``
    (``DecoupledNodeDef``), else none (unnamed, recorded as ``#k``)."""
    if not dataclasses.is_dataclass(defn):
        raise TypeError(
            f"{type(defn).__name__} is not a dataclass def; provenance "
            f"cannot read its name")
    fields = {f.name for f in dataclasses.fields(defn)}
    for field in ("name", "label"):
        if field in fields:
            value = getattr(defn, field)
            return str(value) if value else None
    return None


class _DeclarationsMixin:
    """Base for the composites that record declared intent.

    A subclass lists its stores in :attr:`_DECLARATION_STORES` and
    creates each one as an empty list in ``__init__``.  After that,
    :meth:`_declare` is the only way a def enters a store and
    :meth:`_clear_declarations` the only way the stores are emptied.
    Both invalidate the session's FEMData cache.
    """

    #: Store attribute name -> the def types that store holds.  Matched
    #: on the exact type, so a def type added without a store here is
    #: refused by :meth:`_declare` instead of landing in a sibling's list.
    _DECLARATION_STORES: ClassVar[dict[str, tuple[type, ...]]] = {}
    _parent: Any

    def _declare(self, defn: _D) -> _D:
        """Record ``defn`` and invalidate the FEMData cache.

        1. Route it into the chain-phase broker with
           :func:`try_chain_phase_route`.  That is a no-op before the
           first extraction and for def kinds the router does not
           cover; in a ``from_h5`` session it applies the def or raises.
        2. Append ``defn`` to the store :attr:`_DECLARATION_STORES`
           names for its type.
        3. Bump the session's FEMData counter, so the next
           ``get_fem_data()`` re-extracts instead of returning the
           snapshot taken before ``defn`` existed.
        4. Capture its provenance as ``neutral/<family>/<name|#k>``
           (ADR 0112 D3): one record per user call, so a verb that
           stores several defs records the first.

        Routing comes first so a def the router rejects is never
        stored: the call raised, and the store must not keep what it
        declared.  The router reads only the broker and ``defn``, never
        the store.

        Returns ``defn``, so a verb can end with
        ``return self._declare(defn)``.
        """
        # Function-local: an eager core -> _kernel edge would change the
        # frozen import graph (tests/test_import_dag_polarity.py).
        from apeGmsh._kernel.resolvers._chain_phase_router import (
            try_chain_phase_route,
        )

        attr = self._store_attr_for(defn)
        family = _PROVENANCE_FAMILY[attr]
        store = getattr(self, attr)
        try_chain_phase_route(self._parent, defn)
        store.append(defn)
        self._invalidate_fem()
        _capture_provenance(
            self._parent, "neutral", family, _declaration_name(defn))
        return defn

    def _clear_declarations(self) -> None:
        """Empty every store in :attr:`_DECLARATION_STORES` and
        invalidate the FEMData cache.

        Raises :class:`~apeGmsh.core._compose_errors.ChainPhaseError`
        in a ``from_h5`` / compose session, before anything is emptied.
        There ``get_fem_data()`` returns the broker itself, which holds
        the records the router already applied and those loaded from
        the file; emptying the stores would not retract them.
        """
        if is_kernelless_session(self._parent):
            raise ChainPhaseError(
                f"{type(self).__name__}.clear() cannot retract records "
                f"in a from_h5/compose (chain-phase) session: the "
                f"records already applied to the FEMData broker, and "
                f"those loaded from model.h5, stay in every later "
                f"get_fem_data().  Start again from the saved file with "
                f"apeGmsh.from_h5(path), or remove the declaration in "
                f"the source session and save again."
            )
        for attr in self._DECLARATION_STORES:
            getattr(self, attr).clear()
        self._invalidate_fem()

    def _store_for(self, defn: object) -> list:
        return getattr(self, self._store_attr_for(defn))

    def _store_attr_for(self, defn: object) -> str:
        for attr, kinds in self._DECLARATION_STORES.items():
            if type(defn) in kinds:
                return attr
        raise TypeError(
            f"{type(self).__name__} has no declaration store for "
            f"{type(defn).__name__} — add it to "
            f"{type(self).__name__}._DECLARATION_STORES."
        )

    def _invalidate_fem(self) -> None:
        # Stub parents (test fixtures, ``parent=None``) carry no cache.
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()
