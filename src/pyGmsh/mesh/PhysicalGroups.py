from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import pandas as pd

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tag    = int
DimTag = tuple[int, int]

_DIM_LABEL: dict[int, str] = {
    0: 'points',
    1: 'curves',
    2: 'surfaces',
    3: 'volumes',
}


# ---------------------------------------------------------------------------
# PhysicalGroups — composite class
# ---------------------------------------------------------------------------

class PhysicalGroups:
    """
    Physical-group composite attached to a ``pyGmsh`` instance as
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

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[PhysicalGroups] {msg}")

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def add(
        self,
        dim     : int,
        tags    : list[Tag],
        *,
        name    : str = "",
        tag     : Tag = -1,
    ) -> Tag:
        """
        Create a physical group of dimension *dim* from the entity *tags*.

        Parameters
        ----------
        dim  : entity dimension (0 = points, 1 = curves, 2 = surfaces,
               3 = volumes)
        tags : model-entity tags to include — all must be the same dimension
        name : optional human-readable label assigned immediately
        tag  : requested physical-group tag (``-1`` = auto-assign)

        Returns
        -------
        Tag  the assigned physical-group tag
        """
        pg_tag = gmsh.model.addPhysicalGroup(dim, tags, tag=tag)
        if name:
            gmsh.model.setPhysicalName(dim, pg_tag, name)
        f"{_DIM_LABEL.get(dim, str(dim))} {tags}"
        self._log(
            f"add(dim={dim}, entities={tags}) → pg_tag={pg_tag}"
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
        Return all physical groups as ``(dim, tag)`` pairs.

        Parameters
        ----------
        dim : filter to a single dimension (``-1`` = all)
        """
        return list(gmsh.model.getPhysicalGroups(dim=dim))

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
                g.mesh.set_algorithm(s, "frontal_delaunay_quads")
                g.mesh.set_recombine(s)

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
                    f"No physical group named {name_or_tag!r} at any dimension"
                )
            pg_tag = self.get_tag(dim, name_or_tag)
            if pg_tag is None:
                raise KeyError(
                    f"No physical group named {name_or_tag!r} at dim={dim}"
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
            if gmsh.model.getPhysicalName(dim, pg_tag) == name:
                return pg_tag
        return None

    def summary(self) -> pd.DataFrame:
        """
        Build a DataFrame describing every physical group in the model.

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
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
            name     = gmsh.model.getPhysicalName(dim, pg_tag)
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
            f"get_nodes(dim={dim}, pg={name!r}) → {len(node_tags)} nodes"
        )
        return result

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        groups = gmsh.model.getPhysicalGroups()
        return (
            f"PhysicalGroups(model={self._parent.model_name!r}, "
            f"n_groups={len(groups)})"
        )
