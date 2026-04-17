from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import pandas as pd

from apeGmsh._logging import _HasLogging
from apeGmsh._types import Tag, DimTag
from apeGmsh.core.Labels import is_label_pg

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase

_DIM_LABEL: dict[int, str] = {
    0: 'points',
    1: 'curves',
    2: 'surfaces',
    3: 'volumes',
}


# ---------------------------------------------------------------------------
# PhysicalGroups — composite class
# ---------------------------------------------------------------------------


class PhysicalGroups(_HasLogging):
    """
    Physical-group composite attached to a ``apeGmsh`` instance as
    ``g.physical``.

    Wraps the physical-group subset of the Gmsh Python API with a clean,
    method-chaining interface organised into:

    * **Creation**  — ``add``, ``add_point``, ``add_curve``,
      ``add_surface``, ``add_volume``
    * **Naming**    — ``set_name``, ``remove_name``
    * **Removal**   — ``remove``, ``remove_all``
    * **Queries**   — ``get_all``, ``get_entities``, ``get_groups_for_entity``,
      ``get_name``, ``get_tag``, ``summary``
    * **Mesh nodes**— ``get_nodes``

    All mutating methods return ``self`` for chaining::

        (g.physical
           .add_surface([s_inlet],  name="Inlet")
           .add_surface([s1, s2],   name="Wall")
           .add_volume([vol],       name="Fluid"))

    Parameters
    ----------
    parent : _SessionBase
        The owning instance — used for ``_verbose``.
    """

    _log_prefix = "PhysicalGroups"

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def add(
        self,
        dim     : int,
        tags,
        *,
        name    : str = "",
        tag     : Tag = -1,
    ) -> Tag:
        """Create or append to a physical group.

        If *name* matches an existing PG at this *dim*, the new *tags*
        are merged into it (upsert).  Otherwise a new PG is created.

        Parameters
        ----------
        dim  : entity dimension (0 = points, 1 = curves, 2 = surfaces,
               3 = volumes)
        tags : entity tags — accepts ``int``, ``str`` (label or PG name),
               ``(dim, tag)`` tuples, or a list mixing all of these.
               String references are resolved via ``g.labels`` first,
               then ``g.physical``.
        name : human-readable name.  If a PG with this name already
               exists at *dim*, the new entities are appended.
        tag  : requested physical-group tag (``-1`` = auto-assign).
               Ignored when appending to an existing named PG.

        Returns
        -------
        Tag  the physical-group tag (existing or newly created)
        """
        from typing import cast
        from apeGmsh.core._helpers import resolve_to_tags
        if isinstance(tags, (str, int)):
            tags = [tags]
        # self._parent honours the SessionProtocol structural contract
        # that resolve_to_tags requires — cast to the nominal type
        # mypy expects.
        from apeGmsh._session import _SessionBase
        resolved = resolve_to_tags(
            tags, dim=dim, session=cast(_SessionBase, self._parent),
        )

        # Upsert: if a PG with this name already exists, merge
        if name:
            existing_tag = self.get_tag(dim, name)
            if existing_tag is not None:
                old_ents = list(
                    gmsh.model.getEntitiesForPhysicalGroup(
                        dim, existing_tag))
                combined = sorted(
                    set(old_ents) | set(int(t) for t in resolved))
                gmsh.model.removePhysicalGroups(
                    dimTags=[(dim, existing_tag)])
                pg_tag = gmsh.model.addPhysicalGroup(
                    dim, combined, tag=existing_tag)
                gmsh.model.setPhysicalName(dim, pg_tag, name)
                n_new = len(combined) - len(old_ents)
                self._log(
                    f"add(dim={dim}, name={name!r}): appended "
                    f"{n_new} entity(ies) -> {len(combined)} total")
                return pg_tag

        pg_tag = gmsh.model.addPhysicalGroup(dim, resolved, tag=tag)
        if name:
            gmsh.model.setPhysicalName(dim, pg_tag, name)
        self._log(
            f"add(dim={dim}, entities={tags}) -> pg_tag={pg_tag}"
            + (f", name={name!r}" if name else "")
        )
        return pg_tag

    def add_point(self, tags: list[Tag], *, name: str = "", tag: Tag = -1) -> PhysicalGroups:
        """Shorthand: ``add(dim=0, tags, ...)`` — returns ``self`` for chaining."""
        self.add(0, tags, name=name, tag=tag)
        return self

    def add_curve(self, tags: list[Tag], *, name: str = "", tag: Tag = -1) -> PhysicalGroups:
        """Shorthand: ``add(dim=1, tags, ...)`` — returns ``self`` for chaining."""
        self.add(1, tags, name=name, tag=tag)
        return self

    def add_surface(self, tags: list[Tag], *, name: str = "", tag: Tag = -1) -> PhysicalGroups:
        """Shorthand: ``add(dim=2, tags, ...)`` — returns ``self`` for chaining."""
        self.add(2, tags, name=name, tag=tag)
        return self

    def add_volume(self, tags: list[Tag], *, name: str = "", tag: Tag = -1) -> PhysicalGroups:
        """Shorthand: ``add(dim=3, tags, ...)`` — returns ``self`` for chaining."""
        self.add(3, tags, name=name, tag=tag)
        return self

    # ------------------------------------------------------------------
    # From labels
    # ------------------------------------------------------------------

    def from_label(
        self,
        label_name: str,
        *,
        name: str | None = None,
        dim: int | None = None,
    ) -> Tag:
        """Create (or append to) a PG from a label's entities.

        Parameters
        ----------
        label_name : str
            Label name (without ``_label:`` prefix).
        name : str, optional
            PG name.  Defaults to the label name.
        dim : int, optional
            Dimension.  If ``None``, inferred from the label.

        Returns
        -------
        Tag  the physical-group tag
        """
        labels = getattr(self._parent, 'labels', None)
        if labels is None:
            raise RuntimeError("No labels composite on session.")
        ent_tags = labels.entities(label_name, dim=dim)
        # Infer dim from the label's PG
        if dim is None:
            for d in range(4):
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(d):
                    from apeGmsh.core.Labels import add_prefix
                    if gmsh.model.getPhysicalName(pg_dim, pg_tag) == add_prefix(label_name):
                        dim = pg_dim
                        break
                if dim is not None:
                    break
        if dim is None:
            raise KeyError(
                f"Cannot infer dimension for label {label_name!r}.")
        pg_name = name if name is not None else label_name
        return self.add(dim, ent_tags, name=pg_name)

    def from_labels(
        self,
        label_names: list[str],
        *,
        name: str,
        dim: int | None = None,
    ) -> Tag:
        """Create (or append to) a PG from multiple labels (union).

        Parameters
        ----------
        label_names : list[str]
            Label names to combine.
        name : str
            PG name for the combined group.
        dim : int, optional
            Dimension.  If ``None``, inferred from the first label.

        Returns
        -------
        Tag  the physical-group tag
        """
        labels = getattr(self._parent, 'labels', None)
        if labels is None:
            raise RuntimeError("No labels composite on session.")
        all_tags: list[int] = []
        inferred_dim = dim
        for lbl in label_names:
            ent_tags = labels.entities(lbl, dim=dim)
            all_tags.extend(ent_tags)
            if inferred_dim is None:
                # Infer from first label
                for d in range(4):
                    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(d):
                        from apeGmsh.core.Labels import add_prefix
                        if gmsh.model.getPhysicalName(pg_dim, pg_tag) == add_prefix(lbl):
                            inferred_dim = pg_dim
                            break
                    if inferred_dim is not None:
                        break
        if inferred_dim is None:
            raise KeyError(
                f"Cannot infer dimension for labels {label_names!r}.")
        return self.add(inferred_dim, sorted(set(all_tags)), name=name)

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def set_name(self, dim: int, tag: Tag, name: str) -> PhysicalGroups:
        """
        Assign (or rename) the label of an existing physical group.

        Parameters
        ----------
        dim  : dimension of the physical group
        tag  : physical-group tag
        name : new label
        """
        gmsh.model.setPhysicalName(dim, tag, name)
        self._log(f"set_name(dim={dim}, tag={tag}, name={name!r})")
        return self

    def remove_name(self, name: str) -> PhysicalGroups:
        """
        Remove the name-to-tag mapping for *name*.

        The physical group itself is not deleted — only the name entry.
        """
        gmsh.model.removePhysicalName(name)
        self._log(f"remove_name({name!r})")
        return self

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, dim_tags: list[DimTag]) -> PhysicalGroups:
        """
        Remove specific physical groups.

        Parameters
        ----------
        dim_tags : ``[(dim, pg_tag), ...]`` pairs identifying groups to remove
        """
        gmsh.model.removePhysicalGroups(dimTags=dim_tags)
        self._log(f"remove({dim_tags})")
        return self

    def remove_all(self) -> PhysicalGroups:
        """Remove every physical group in the current model."""
        gmsh.model.removePhysicalGroups()
        self._log("remove_all()")
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all(self, dim: int = -1) -> list[DimTag]:
        """
        Return all **user-facing** physical groups as ``(dim, tag)``
        pairs.  Internal label PGs (Tier 1 naming, prefixed with
        ``_label:``) are filtered out — use ``g.labels.get_all()``
        to see those.

        Parameters
        ----------
        dim : filter to a single dimension (``-1`` = all)
        """
        return [
            (d, t) for d, t in gmsh.model.getPhysicalGroups(dim=dim)
            if not is_label_pg(gmsh.model.getPhysicalName(d, t))
        ]

    def get_entities(self, dim: int, tag: Tag) -> list[Tag]:
        """
        Return the model-entity tags contained in a physical group.

        Parameters
        ----------
        dim : dimension of the physical group
        tag : physical-group tag
        """
        return list(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))

    def entities(
        self,
        name_or_tag,
        *,
        dim: int | None = None,
    ) -> list[Tag]:
        """
        Resolve a physical group to its entity tags — by **name** or by tag.

        This is the convenience entry point used by mesh commands that want
        to act on "everything in the 'Concrete' PG" without the caller
        having to chain ``get_tag`` + ``get_entities`` manually.

        Parameters
        ----------
        name_or_tag : str | int
            Physical group name, or the raw PG tag (in which case ``dim``
            is required).
        dim : int | None, optional
            Dimension to search.  If omitted and ``name_or_tag`` is a
            string, all dimensions are searched from 0 to 3 and the first
            match wins.  Required when ``name_or_tag`` is an int.

        Returns
        -------
        list[Tag]
            Flat list of model-entity tags contained in the group.

        Raises
        ------
        KeyError
            If no physical group with that name exists.
        TypeError
            If ``name_or_tag`` is an int but ``dim`` is not provided.

        Examples
        --------
        ::

            # Fan a per-surface algorithm out over every surface in a PG
            for s in g.physical.entities("Concrete", dim=2):
                g.mesh.generation.set_algorithm(s, "frontal_delaunay_quads")
                g.mesh.structured.set_recombine(s)

            # Feed a distance field directly from a PG
            joint = g.physical.entities("BeamColumnJoint", dim=2)
            d = g.mesh.field.distance(surfaces=joint)
        """
        if isinstance(name_or_tag, str):
            if dim is None:
                for d in (0, 1, 2, 3):
                    pg_tag = self.get_tag(d, name_or_tag)
                    if pg_tag is not None:
                        return list(gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag))
                raise KeyError(
                    f"No physical group named {name_or_tag!r} at any "
                    f"dimension.  If this is a label, use "
                    f"g.labels.entities({name_or_tag!r}) or promote it "
                    f"with g.labels.promote_to_physical({name_or_tag!r})."
                )
            pg_tag = self.get_tag(dim, name_or_tag)
            if pg_tag is None:
                raise KeyError(
                    f"No physical group named {name_or_tag!r} at dim={dim}.  "
                    f"If this is a label, use g.labels.entities() or "
                    f"g.labels.promote_to_physical()."
                )
            return list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))

        # int / PG tag path
        if dim is None:
            raise TypeError(
                "entities(): when passing a raw PG tag, `dim` must be given"
            )
        return list(gmsh.model.getEntitiesForPhysicalGroup(dim, int(name_or_tag)))

    def get_groups_for_entity(self, dim: int, tag: Tag) -> list[Tag]:
        """
        Return the physical-group tags that contain a given model entity.

        Parameters
        ----------
        dim : entity dimension
        tag : model-entity tag
        """
        return list(gmsh.model.getPhysicalGroupsForEntity(dim, tag))

    def get_name(self, dim: int, tag: Tag) -> str:
        """
        Return the name of a physical group, or ``""`` if unnamed.

        Parameters
        ----------
        dim : dimension of the physical group
        tag : physical-group tag
        """
        return gmsh.model.getPhysicalName(dim, tag)

    def get_tag(self, dim: int, name: str) -> Tag | None:
        """
        Look up the tag of a named physical group.

        Returns ``None`` if no group with that name and dimension exists.

        Parameters
        ----------
        dim  : dimension to search
        name : human-readable label
        """
        for _, pg_tag in gmsh.model.getPhysicalGroups(dim=dim):
            pg_name = gmsh.model.getPhysicalName(dim, pg_tag)
            if is_label_pg(pg_name):
                continue
            if pg_name == name:
                return pg_tag
        return None

    def summary(self) -> pd.DataFrame:
        """
        Build a DataFrame describing every **user-facing** physical
        group in the model.  Internal label PGs are excluded.

        Returns
        -------
        pd.DataFrame  indexed by ``(dim, pg_tag)`` with columns:

        ``dim``           entity dimension
        ``pg_tag``        physical-group tag
        ``name``          label (empty string if unnamed)
        ``n_entities``    number of model entities in the group
        ``entity_tags``   comma-separated entity tags as a string
        """
        rows: list[dict] = []
        for dim, pg_tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, pg_tag)
            if is_label_pg(name):
                continue
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
            rows.append({
                'dim'        : dim,
                'pg_tag'     : pg_tag,
                'name'       : name,
                'n_entities' : len(entities),
                'entity_tags': ", ".join(str(t) for t in entities),
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'pg_tag', 'name', 'n_entities', 'entity_tags']
            )

        df = (
            pd.DataFrame(rows)
            .set_index(['dim', 'pg_tag'])
            .sort_index()
        )

        if self._parent._verbose:
            print("\n--- Physical Groups ---")
            print(df.to_string())

        return df

    # ------------------------------------------------------------------
    # Mesh nodes
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        dim: int,
        tag: Tag,
    ) -> dict:
        """
        Return the mesh nodes belonging to a physical group.

        Must be called after mesh generation.

        Parameters
        ----------
        dim : dimension of the physical group
        tag : physical-group tag

        Returns
        -------
        dict
            ``'tags'``   : ndarray(N,)   — node tags
            ``'coords'`` : ndarray(N, 3) — XYZ coordinates
        """
        import numpy as np
        node_tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
        result = {
            'tags'  : np.array(node_tags, dtype=np.int64),
            'coords': np.array(coords).reshape(-1, 3),
        }
        name = gmsh.model.getPhysicalName(dim, tag) or str(tag)
        self._log(
            f"get_nodes(dim={dim}, pg={name!r}) -> {len(node_tags)} nodes"
        )
        return result

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        groups = gmsh.model.getPhysicalGroups()
        return (
            f"PhysicalGroups(model={self._parent.name!r}, "
            f"n_groups={len(groups)})"
        )
