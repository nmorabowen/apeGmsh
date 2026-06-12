"""
_Recipe — one-call unstructured / structured meshing (ADR 0059).

Accessed via ``g.mesh.recipe``.  Pure orchestration over the sibling
sub-composites (``sizing`` / ``field`` / ``structured`` / ``generation``)
— no new gmsh state of its own beyond the region size fields it builds
through ``g.mesh.field``.

Two verbs::

    # easy model — one line, generates immediately
    g.mesh.recipe.unstructured(min_size=0.2, max_size=1.0)

    # mixed model — compose region recipes, then generate once
    g.mesh.recipe.structured("soil_block", size=2.0, recombine=False)
    g.mesh.recipe.unstructured("tunnel_liner", max_size=0.4)
    g.mesh.generation.generate(dim=3)

Region sizing is implemented with gmsh ``Constant`` fields combined
through a single recipe-owned ``Min`` background field — **not** with
per-point characteristic lengths, which bleed across shared corner
points (ADR 0059 §3).
"""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh

from apeGmsh._types import DimTag

#: gmsh's own "no size constraint" sentinel (default Mesh.MeshSizeMax).
_NO_SIZE = 1e22

#: Zero-argument heuristic — bbox diagonal / this (ADR 0059 §3, Open Q2).
_DEFAULT_DIAG_FRACTION = 20.0

_FALLBACK_MODES = ("unstructured", "warn", "strict")


class MeshRecipeError(RuntimeError):
    """A mesh recipe cannot proceed.

    Raised by ``g.mesh.recipe`` for: a structured recipe with
    ``fallback="strict"`` hitting non-decomposable entities, the mixed
    quad/tri interface guard (a recombined-structured volume sharing a
    face with an unstructured neighbor — gmsh has no pyramid
    transition), or a recipe target that resolves to nothing sizeable.
    """


class _Recipe:
    """One-call unstructured / structured meshing recipes (ADR 0059)."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh
        # Constant fields created by targeted recipes, combined through
        # one recipe-owned Min background field.
        self._region_field_tags: list[int] = []
        # User-authored background fields folded into the Min combiner
        # (ADR 0059 Open Q3 — fold + log, never silently replace).
        self._folded_external_tags: list[int] = []
        self._min_field_tag: int | None = None

    def _guard(self, op: str) -> None:
        """Phase 3B.2d / ADR 0038 — chain-phase freeze guard."""
        from apeGmsh.core._compose_errors import chain_phase_guard
        chain_phase_guard(self._mesh._parent, f"g.mesh.recipe.{op}")

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------

    def unstructured(
        self,
        target=None,
        *,
        max_size: float | None = None,
        min_size: float = 0.0,
        dim: int | None = None,
        generate: bool | None = None,
    ) -> "_Recipe":
        """Unstructured-mesh recipe — size band in, mesh out.

        Parameters
        ----------
        target :
            ``None`` (default) — the whole model.  Otherwise a label,
            physical-group name, part label, ``(dim, tag)`` dimtag, or
            a list of any mix; resolution order label → PG → part.
        max_size : float, optional
            Target element size.  Whole-model: the global ceiling
            (``Mesh.MeshSizeMax``).  Targeted: the size inside the
            target region (a ``Constant`` field).  ``None`` derives a
            default from the model bounding-box diagonal
            (``diag / 20``), so ``g.mesh.recipe.unstructured()`` with
            no arguments produces something sane on any model.
        min_size : float, default ``0.0``
            Element-size floor.  Whole-model: ``Mesh.MeshSizeMin``.
            Targeted: the floor is global by gmsh construction — a
            nonzero value here only ever *lowers* the current global
            floor (never raises it, which would affect other regions).
        dim : int, optional
            Mesh dimension for generation.  ``None`` = highest
            dimension present in the model.
        generate : bool, optional
            ``None`` (default) resolves by scope: whole-model recipes
            generate immediately; targeted recipes only declare, so
            several compose before one
            ``g.mesh.generation.generate(...)``.  Explicit
            ``True``/``False`` always wins.

        Notes
        -----
        The whole-model form disables size inheritance from CAD points
        and curvature (``set_size_sources(from_points=False,
        from_curvature=False)``) — imported STEP/IGES files bake
        per-point characteristic lengths that would silently override
        the requested band.  Call ``g.mesh.sizing.set_size_sources``
        afterwards to opt back in.

        Targeted region sizing uses a ``Constant`` field combined
        through a recipe-owned ``Min`` background field — not per-point
        characteristic lengths, which bleed across shared corner
        points.
        """
        self._guard("unstructured")
        if max_size is None:
            max_size = self._default_max_size()
        if max_size <= 0.0:
            raise ValueError(f"unstructured: max_size must be > 0, got {max_size}")
        if min_size < 0.0 or min_size > max_size:
            raise ValueError(
                f"unstructured: min_size must be in [0, max_size], "
                f"got min_size={min_size}, max_size={max_size}"
            )

        if target is None:
            self._mesh.sizing.set_size_sources(
                from_points=False, from_curvature=False,
            )
            self._mesh.sizing.set_global_size(max_size, min_size)
        else:
            dimtags = self._resolve_target(target, what="unstructured")
            self._add_region_field(dimtags, max_size, source_ref=repr(target))
            self._lower_floor_if_clamping(max_size, min_size)

        self._mesh._directives.append({
            'kind': 'recipe_unstructured',
            'target': None if target is None else repr(target),
            'max_size': max_size, 'min_size': min_size,
        })
        self._mesh._log(
            f"recipe.unstructured(target={target!r}, max_size={max_size}, "
            f"min_size={min_size})"
        )

        if self._should_generate(generate, target):
            self._generate(dim)
        return self

    def structured(
        self,
        target=None,
        *,
        size=None,
        n=None,
        recombine: bool = True,
        fallback: str = "unstructured",
        dim: int | None = None,
        generate: bool | None = None,
    ) -> "_Recipe":
        """Structured-mesh recipe — transfinite cascade + sizing fallback.

        Delegates the constraint cascade to
        :meth:`~apeGmsh.mesh._mesh_structured._Structured.set_transfinite`
        (same ``n=`` / ``size=`` grammar: scalar, per-axis dict, or
        per-principal-axis tuple), then adds what the dispatcher
        deliberately does not do: sizing fallback for skipped entities,
        generation, and the mixed-interface guard at generate time.

        Parameters
        ----------
        target :
            ``None`` (default) — every volume in the model (every
            surface in a 2-D model).  Otherwise any flexible reference
            (label / PG / part / dimtag / list).
        size, n :
            Exactly one.  Same grammar as ``set_transfinite``.
        recombine : bool, default ``True``
            Quads on faces / hexes in volumes.  ``False`` gives a
            structured tri/tet mesh — required when a structured region
            shares faces with an unstructured neighbor (see the
            mixed-interface guard, :meth:`check`).
        fallback : str, default ``"unstructured"``
            What happens to entities the cascade warn-skips
            (non-hex/quad-decomposable):

            - ``"unstructured"`` — the skipped entity gets a region
              size field at an equivalent size (``size`` form: that
              size; ``n`` form: characteristic edge length divided by
              ``n - 1``).  You always get a mesh.
            - ``"warn"`` — exactly ``set_transfinite``'s behavior
              (warn + skip; whatever global sizing applies).
            - ``"strict"`` — raise :class:`MeshRecipeError` listing
              the offending entities.
        dim, generate :
            As in :meth:`unstructured`.
        """
        self._guard("structured")
        if fallback not in _FALLBACK_MODES:
            raise ValueError(
                f"structured: fallback must be one of {_FALLBACK_MODES}, "
                f"got {fallback!r}"
            )

        # Whole-model on a 2-D geometry: set_transfinite(None) resolves
        # to dim-3 entities only, which would silently do nothing.
        st_target = target
        if target is None and not gmsh.model.getEntities(3):
            st_target = gmsh.model.getEntities(2)

        st = self._mesh.structured
        st.set_transfinite(st_target, n=n, size=size, recombine=recombine)
        skipped: list[DimTag] = list(st._last_skipped)

        if skipped:
            if fallback == "strict":
                listing = ", ".join(f"(dim={d}, tag={t})" for d, t in skipped)
                raise MeshRecipeError(
                    f"recipe.structured(fallback='strict'): "
                    f"{len(skipped)} entit{'y is' if len(skipped) == 1 else 'ies are'} "
                    f"not transfinite-decomposable: {listing}. See the "
                    f"set_transfinite warnings above for the per-entity "
                    f"reason (edge-direction cluster counts)."
                )
            if fallback == "unstructured":
                for dt in skipped:
                    fb_size = self._fallback_size(dt, n=n, size=size)
                    self._add_region_field(
                        [dt], fb_size, source_ref=f"structured-fallback{dt}",
                    )
                    self._lower_floor_if_clamping(fb_size, 0.0)
                    self._mesh._log(
                        f"recipe.structured: entity {dt} not decomposable — "
                        f"unstructured fallback at size={fb_size:.6g}"
                    )

        self._mesh._directives.append({
            'kind': 'recipe_structured',
            'target': None if target is None else repr(target),
            'size': size, 'n': n, 'recombine': recombine,
            'fallback': fallback,
            'skipped': [list(dt) for dt in skipped],
        })
        self._mesh._log(
            f"recipe.structured(target={target!r}, size={size!r}, n={n!r}, "
            f"recombine={recombine}, fallback={fallback!r}, "
            f"skipped={len(skipped)})"
        )

        if self._should_generate(generate, target):
            self._generate(dim)
        return self

    # ------------------------------------------------------------------
    # Mixed-interface guard (ADR 0059 §5)
    # ------------------------------------------------------------------

    def check(self) -> "_Recipe":
        """Run the mixed quad/tri interface guard without generating.

        A transfinite + recombined volume puts **quads** on its
        boundary faces; a conformal unstructured (tet) neighbor needs
        **triangles** there, and gmsh has no robust automatic pyramid
        transition.  This guard fails loud at declaration time instead
        of letting ``generate()`` trip over it mid-mesh.

        The classification is closed-world over apeGmsh directives
        (``set_transfinite*`` / ``set_recombine`` calls made through
        ``g.mesh.structured`` or the recipes); constraints applied
        through raw ``gmsh.model.mesh.*`` calls are invisible to it.
        It runs automatically when a *recipe* generates — never inside
        the raw ``g.mesh.generation.generate()`` path.

        Raises
        ------
        MeshRecipeError
            Naming each shared face and the two remediations:
            ``recombine=False`` on the structured side (transfinite
            prisms/tets conform to triangles), or make the neighbor
            structured too.
        """
        conflicts = self._find_mixed_interfaces()
        if conflicts:
            lines = [
                f"  face (2, {face}) is shared by recombined-structured "
                f"volume {a} and unstructured volume {b}"
                for face, a, b in conflicts
            ]
            raise MeshRecipeError(
                "Mixed structured/unstructured interface(s): a "
                "transfinite + recombined volume puts QUADS on its "
                "boundary faces, but a conformal unstructured neighbor "
                "needs TRIANGLES there (gmsh has no pyramid "
                "transition):\n" + "\n".join(lines) + "\n"
                "Fix: pass recombine=False on the structured side "
                "(structured prisms/tets conform to triangles), or make "
                "the neighbor structured + recombined too."
            )
        return self

    def _find_mixed_interfaces(self) -> list[tuple[int, int, int]]:
        """Return ``(face_tag, structured_vol, unstructured_vol)`` triples."""
        transfinite_vols: set[int] = set()
        recombined_faces: set[int] = set()
        for d in self._mesh._directives:
            if d.get('kind') == 'transfinite_volume':
                transfinite_vols.add(int(d['tag']))
            elif d.get('kind') == 'recombine' and int(d.get('dim', 2)) == 2:
                recombined_faces.add(int(d['tag']))

        if not transfinite_vols:
            return []

        vol_faces: dict[int, set[int]] = {}
        for _, vtag in gmsh.model.getEntities(3):
            bnd = gmsh.model.getBoundary(
                [(3, vtag)], combined=False, oriented=False, recursive=False,
            )
            vol_faces[int(vtag)] = {abs(int(t)) for dd, t in bnd if dd == 2}

        hex_structured = {
            v for v in vol_faces
            if v in transfinite_vols and vol_faces[v] & recombined_faces
        }
        others = set(vol_faces) - hex_structured
        conflicts: list[tuple[int, int, int]] = []
        for a in sorted(hex_structured):
            for b in sorted(others):
                for face in sorted(vol_faces[a] & vol_faces[b]):
                    conflicts.append((face, a, b))
        return conflicts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _should_generate(generate: bool | None, target) -> bool:
        # ADR 0059 §2 — auto by scope: whole-model recipes generate,
        # targeted recipes compose.  Explicit True/False always wins.
        if generate is not None:
            return generate
        return target is None

    def _generate(self, dim: int | None) -> None:
        if dim is None:
            dim = 3 if gmsh.model.getEntities(3) else (
                2 if gmsh.model.getEntities(2) else 1
            )
        if dim == 3:
            self.check()
        self._mesh.generation.generate(dim=dim)

    def _resolve_target(self, target, *, what: str) -> list[DimTag]:
        from apeGmsh.core._helpers import resolve_to_dimtags
        dimtags = resolve_to_dimtags(
            target, default_dim=3, session=self._mesh._parent,
        )
        if not dimtags:
            raise MeshRecipeError(
                f"recipe.{what}: target {target!r} resolved to no entities."
            )
        return dimtags

    @staticmethod
    def _default_max_size() -> float:
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
        except Exception as e:  # gmsh raises bare Exception on empty models
            raise MeshRecipeError(
                "recipe: cannot derive a default element size — the model "
                "bounding box is empty. Add geometry first, or pass "
                "max_size= explicitly."
            ) from e
        diag = math.sqrt(
            (bb[3] - bb[0]) ** 2 + (bb[4] - bb[1]) ** 2 + (bb[5] - bb[2]) ** 2
        )
        if not math.isfinite(diag) or diag <= 0.0:
            raise MeshRecipeError(
                "recipe: cannot derive a default element size — the model "
                "bounding box is empty. Add geometry first, or pass "
                "max_size= explicitly."
            )
        return diag / _DEFAULT_DIAG_FRACTION

    def _add_region_field(
        self,
        dimtags: list[DimTag],
        size: float,
        *,
        source_ref: str,
    ) -> None:
        """Constant field (VIn=size inside the entities) + Min refresh."""
        by_dim: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for d, t in dimtags:
            by_dim[d].append(int(t))

        f = self._mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(f, "VIn", size)
        gmsh.model.mesh.field.setNumber(f, "VOut", _NO_SIZE)
        gmsh.model.mesh.field.setNumber(f, "IncludeBoundary", 1)
        if by_dim[3]:
            gmsh.model.mesh.field.setNumbers(f, "VolumesList", by_dim[3])
        if by_dim[2]:
            gmsh.model.mesh.field.setNumbers(f, "SurfacesList", by_dim[2])
        if by_dim[1]:
            gmsh.model.mesh.field.setNumbers(f, "CurvesList", by_dim[1])
        if by_dim[0]:
            gmsh.model.mesh.field.setNumbers(f, "PointsList", by_dim[0])

        self._region_field_tags.append(f)
        self._mesh._directives.append({
            'kind': 'recipe_region_field', 'field_tag': f,
            'size': size, 'source_ref': source_ref,
            'dimtags': [list(dt) for dt in dimtags],
        })
        self._refresh_min_field()

    def _refresh_min_field(self) -> None:
        """(Re)point the recipe-owned Min background at all region fields.

        A user-authored background field (set through
        ``g.mesh.field.set_background``) is folded into the Min
        combiner rather than silently replaced (ADR 0059 Open Q3).
        """
        user_bg = self._mesh._background_field_tag
        if (
            user_bg is not None
            and user_bg != self._min_field_tag
            and user_bg not in self._region_field_tags
            and user_bg not in self._folded_external_tags
        ):
            self._folded_external_tags.append(user_bg)
            self._mesh._log(
                f"recipe: folding existing background field {user_bg} "
                f"into the recipe Min combiner"
            )

        tags = self._region_field_tags + self._folded_external_tags
        if self._min_field_tag is None:
            self._min_field_tag = self._mesh.field.minimum(tags)
        else:
            gmsh.model.mesh.field.setNumbers(
                self._min_field_tag, "FieldsList", [float(t) for t in tags],
            )
        self._mesh.field.set_background(self._min_field_tag)

    def _lower_floor_if_clamping(self, region_size: float, min_size: float) -> None:
        """Widen the global clamp band so a region field survives it.

        ``Mesh.MeshSizeMin`` / ``MeshSizeMax`` clamp every size source,
        including background fields.  The floor is only ever *lowered*
        (a raise would coarsen other regions); a ceiling below the
        requested region size is warned about, not raised — the global
        ceiling is the user's stated intent everywhere else.
        """
        floor = gmsh.option.getNumber("Mesh.MeshSizeMin")
        new_floor = floor
        if floor > region_size:
            new_floor = region_size
        if min_size > 0.0 and min_size < new_floor:
            new_floor = min_size
        if new_floor != floor:
            gmsh.option.setNumber("Mesh.MeshSizeMin", new_floor)
            self._mesh._log(
                f"recipe: lowered global Mesh.MeshSizeMin {floor} -> "
                f"{new_floor} so the region field is not clamped"
            )
        ceiling = gmsh.option.getNumber("Mesh.MeshSizeMax")
        if ceiling < region_size:
            warnings.warn(
                f"recipe: requested region size {region_size} exceeds the "
                f"global Mesh.MeshSizeMax ({ceiling}); the global ceiling "
                f"wins. Raise it via g.mesh.sizing.set_global_size if the "
                f"coarser region size is intended.",
                UserWarning,
                stacklevel=3,
            )

    def _fallback_size(self, dt: DimTag, *, n, size) -> float:
        """Equivalent unstructured size for a skipped entity (ADR §4).

        ``size`` form: the finest requested value.  ``n`` form: the
        entity's mean boundary-curve length divided by ``(n_eff - 1)``
        with ``n_eff`` the finest (largest) requested count.
        """
        def _values(spec) -> list[float]:
            if isinstance(spec, dict):
                return [float(v) for v in spec.values()]
            if isinstance(spec, (tuple, list)):
                return [float(v) for v in spec]
            return [float(spec)]

        if size is not None:
            return min(_values(size))

        n_eff = max(int(v) for v in _values(n))
        if n_eff < 2:
            n_eff = 2
        d, t = dt
        if d == 3:
            curves = self._mesh._parent.model.queries.boundary_curves(t)
            curve_dts = [(int(cd), int(ct)) for cd, ct in curves]
        else:
            bnd = gmsh.model.getBoundary(
                [(d, t)], combined=False, oriented=False, recursive=False,
            )
            curve_dts = [(dd, abs(int(tt))) for dd, tt in bnd if dd == 1]
        lengths: list[float] = []
        for cdt in curve_dts:
            bb = gmsh.model.getBoundingBox(*cdt)
            lengths.append(math.sqrt(
                (bb[3] - bb[0]) ** 2
                + (bb[4] - bb[1]) ** 2
                + (bb[5] - bb[2]) ** 2
            ))
        if not lengths:
            bb = gmsh.model.getBoundingBox(d, t)
            lengths = [math.sqrt(
                (bb[3] - bb[0]) ** 2
                + (bb[4] - bb[1]) ** 2
                + (bb[5] - bb[2]) ** 2
            )]
        mean_len = sum(lengths) / len(lengths)
        return mean_len / (n_eff - 1)
