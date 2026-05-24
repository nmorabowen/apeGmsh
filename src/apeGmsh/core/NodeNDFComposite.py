"""
NodeNDFComposite -- Declare per-node ``ndf`` (DOF count) overrides.

Explicit-only API: every node that needs an ``ndf`` must be covered
by an :meth:`set` call (targeted at a label / PG / part / mesh
selection) or by :meth:`set_default` (blanket for everything else).
Nodes not covered raise from :meth:`NodeComposite.ndf_for`.

apeGmsh deliberately does **not** infer ``ndf`` from element class.
A prior attempt (PR #307) defaulted line/surface elements to 6 and
volume elements to 3; the resulting silent rewrites broke
fragment_all() pipelines and shell-on-solid mixed meshes in
non-obvious ways.  The user is the single source of truth.

Pipeline
--------
1. **Declare** (pre-mesh): :meth:`set` / :meth:`set_default` store
   :class:`~apeGmsh._kernel.defs.node_ndf.NodeNDFDef` objects on
   ``self._defs``.
2. **Resolve** (at FEM-build time): ``_fem_factory._populate_node_ndf``
   walks the def list in declaration order, resolves each target via
   the shared loads/masses resolver, and writes ``ndf`` values into
   the per-node ``int8`` array on :class:`NodeComposite`.  Default
   (if declared) fills every node still at the sentinel.

Targets follow the same flexible scheme as
:class:`LoadsComposite` / :class:`MassesComposite`:

* a list of ``(dim, tag)`` tuples
* a part label (``g.parts.instances[label]``)
* a physical group name (``g.physical``)
* a mesh-selection name (``g.mesh_selection``)
* a Tier 1 internal label
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh._kernel.defs.node_ndf import NodeNDFDef


_VALID_NDF_RANGE = range(1, 7)  # [1, 6] inclusive


def _validate_ndf(ndf) -> int:
    """Coerce + validate an ``ndf`` value (typed int in [1, 6])."""
    if isinstance(ndf, bool) or not isinstance(ndf, int):
        raise TypeError(
            f"ndf must be an int in [1, 6]; got {type(ndf).__name__} "
            f"{ndf!r}"
        )
    if ndf not in _VALID_NDF_RANGE:
        raise ValueError(
            f"ndf must be in [1, 6]; got {ndf}"
        )
    return int(ndf)


class NodeNDFComposite:
    """Explicit per-node ``ndf`` (DOF count) declarations.

    Sibling to :class:`g.constraints` / :class:`g.loads` /
    :class:`g.masses` — declare on geometry / PG / label / part /
    mesh-selection before meshing; the broker materialises the
    per-node ``ndf`` vector at :meth:`Mesh.queries.get_fem_data` time.

    Examples
    --------
    Uniform 3D solid model::

        g.node_ndf.set_default(ndf=3)

    Shell-on-solid coupling::

        g.node_ndf.set_default(ndf=3)        # everything is solid by default
        g.node_ndf.set('ShellFace', ndf=6)   # shell nodes get rotational DOFs

    Truss-only model — no default needed when every node is in a PG::

        g.node_ndf.set('Trusses', ndf=3)

    Inspection::

        len(g.node_ndf)          # number of registered defs
        list(g.node_ndf)         # iterate NodeNDFDef objects
        g.node_ndf.list()        # same; explicit name
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        # Declaration-ordered defs; last matching def wins per node.
        self._defs: list[NodeNDFDef] = []
        # Index of the default def in ``self._defs`` (set via
        # :meth:`set_default`); replaced on re-call, not appended.
        # ``None`` means no default declared — orphan / uncovered
        # nodes raise from ``NodeComposite.ndf_for``.
        self._default_idx: int | None = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def set(
        self,
        target,
        *,
        ndf: int,
        name: str | None = None,
    ) -> NodeNDFDef:
        """Declare the ``ndf`` for every node resolved from *target*.

        Resolution happens at FEM-build time
        (``g.mesh.queries.get_fem_data()``) via the same
        label-then-PG-then-part precedence chain that loads and
        masses use.  Targets unresolved at extraction raise
        ``KeyError`` (matches the dimensional resolution contract).

        Multiple :meth:`set` calls may target overlapping node sets;
        the last matching def wins.

        Parameters
        ----------
        target :
            Label name, physical-group name, part label, raw
            ``[(dim, tag), ...]`` list, or mesh-selection name —
            anything the shared resolver accepts.
        ndf : int, keyword-only
            DOF count for every resolved node.  Must be in ``[1, 6]``.
        name : str, optional
            Friendly tag stored on the def for debugging.

        Returns
        -------
        NodeNDFDef
            The stored def (mostly useful for tests / introspection).
        """
        ndf_int = _validate_ndf(ndf)
        self._warn_if_post_extraction("set")
        defn = NodeNDFDef(target=target, ndf=ndf_int, name=name)
        self._defs.append(defn)
        return defn

    def set_default(
        self,
        *,
        ndf: int,
        name: str | None = None,
    ) -> NodeNDFDef:
        """Declare the fallback ``ndf`` for every node not covered by
        an explicit :meth:`set` call.

        Re-calling replaces the existing default (does not append a
        second default).  Calling :meth:`set` after :meth:`set_default`
        with overlapping nodes still wins for those nodes — the
        default only fills slots that remain at the sentinel after
        every targeted def has been resolved.

        Parameters
        ----------
        ndf : int, keyword-only
            DOF count for the fallback.  Must be in ``[1, 6]``.
        name : str, optional
            Friendly tag stored on the def for debugging.

        Returns
        -------
        NodeNDFDef
            The stored default def.
        """
        ndf_int = _validate_ndf(ndf)
        self._warn_if_post_extraction("set_default")
        defn = NodeNDFDef(target=None, ndf=ndf_int, name=name)
        if self._default_idx is None:
            self._defs.append(defn)
            self._default_idx = len(self._defs) - 1
        else:
            self._defs[self._default_idx] = defn
        return defn

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list(self) -> list[NodeNDFDef]:
        """Return the registered defs in declaration order.

        Includes the default def (if declared); identify it by
        ``defn.target is None``.
        """
        return list(self._defs)

    def clear(self) -> None:
        """Drop every registered def (including the default).

        Symmetric with :meth:`set` / :meth:`set_default`: warns if
        called after a ``get_fem_data()`` build because the cached
        broker still holds the pre-clear ndf array; re-extract to
        propagate the wipe.
        """
        self._warn_if_post_extraction("clear")
        self._defs.clear()
        self._default_idx = None

    # ------------------------------------------------------------------
    # Internal — used by the resolver path
    # ------------------------------------------------------------------

    def _default_def(self) -> NodeNDFDef | None:
        """Return the default def, or ``None`` if none was declared."""
        if self._default_idx is None:
            return None
        return self._defs[self._default_idx]

    def _targeted_defs(self) -> list[NodeNDFDef]:
        """Return every non-default def in declaration order."""
        if self._default_idx is None:
            return list(self._defs)
        return [d for i, d in enumerate(self._defs) if i != self._default_idx]

    def _warn_if_post_extraction(self, method_name: str) -> None:
        """Warn when defs are mutated after the broker has been built.

        The broker caches per-node ``ndf`` arrays at
        ``get_fem_data()`` time; later ``set`` / ``set_default`` /
        ``clear`` calls do not retroactively rewrite an
        already-extracted FEM.  Re-extract the broker if the new
        declaration must take effect.

        The flag is *cleared* after warning so that a batch of
        post-extract mutations only warns on the first call; the next
        ``get_fem_data()`` re-stamps the flag, restoring the guard for
        the next round.  Without this, the reset-at-top-of-build trick
        from PR #317 fired spurious warnings during the legitimate
        re-extract sequence ``extract → set → extract → set`` (the
        polish commit only suppressed warnings *during* a build, not
        the warning that fires *between* two builds when the new
        declaration will in fact be picked up by the next extract).
        """
        if getattr(self._parent, "_fem_built", False):
            warnings.warn(
                f"g.node_ndf.{method_name}() called after get_fem_data() — "
                f"the existing FEMData snapshot is cached and will not see "
                f"this declaration; re-run g.mesh.queries.get_fem_data(...) "
                f"to refresh.",
                UserWarning,
                stacklevel=3,
            )
            # Clear the flag so the user can finish their batch of
            # re-declarations without N redundant warnings.  Next
            # ``get_fem_data()`` re-stamps it.
            try:
                self._parent._fem_built = False
            except AttributeError:
                pass  # not a vanilla session — skip silently

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._defs)

    def __iter__(self):
        return iter(self._defs)

    def __bool__(self) -> bool:
        return bool(self._defs)

    def __repr__(self) -> str:
        if not self._defs:
            return "NodeNDFComposite(empty)"
        n_targeted = len(self._targeted_defs())
        default = self._default_def()
        default_str = (
            f", default ndf={default.ndf}" if default is not None else ""
        )
        return (
            f"NodeNDFComposite({n_targeted} targeted def(s){default_str})"
        )


__all__ = ["NodeNDFComposite"]
