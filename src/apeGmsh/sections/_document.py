"""``SectionDocument`` — declarative section documents (ADR 0080 B1).

A versioned JSON document that fully describes a section; the source
of truth for the builder GUI and the public headless API (the parity
law: every GUI action is a document mutation). B1 ships the
**continuum lane**: parametric shapes + freehand polygons + booleans
+ per-region materials + mesh prefs, built headlessly into a
:class:`~apeGmsh.sections.SectionProperties`. The fiber lane (RC
templates) is B2 — ``kind="fiber"`` documents are rejected with
guidance until then.

Versioning: ``SECTION_DOC_VERSION`` follows the ADR 0023 additive-minor
law with the corrected (#836) window direction — this loader opens
documents at its own minor and the previous minor; a loader older than
the document refuses it loudly.

The document deliberately owns the composite-partition law: the
``embed`` boolean op is the one-step "inner region inside an outer
region" primitive (cut with ``remove_tool=False``, then
``fragment_pair``) — the double-cover trap cannot be authored through
it. Raw ``cut`` / ``fragment_pair`` steps remain available for
non-overlapping compositions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from ._materials import SectionMaterial

if TYPE_CHECKING:  # pragma: no cover
    from ._analysis import SectionProperties


__all__ = ["SECTION_DOC_VERSION", "SectionDocument", "SectionDocumentError"]


#: Document schema version (ADR 0080). Additive-minor law: this loader
#: accepts documents at the same minor and the previous minor of the
#: same major; anything newer or older refuses loudly.
SECTION_DOC_VERSION: str = "1.0.0"

#: Parametric shapes the continuum lane accepts, mapped to their
#: ``g.sections.*`` builder names and required parameter keys.
_SHAPE_PARAMS: dict[str, tuple[str, ...]] = {
    "W_face": ("bf", "tf", "h", "tw"),
    "rect_face": ("b", "h"),
    "rect_hollow_face": ("b", "h", "t"),
    "pipe_face": ("r",),
    "pipe_hollow_face": ("r", "t"),
    "angle_face": ("b", "h", "t"),
    "channel_face": ("bf", "tf", "h", "tw"),
    "tee_face": ("bf", "tf", "h", "tw"),
}

_MATERIAL_KEYS = ("E", "nu", "G", "fy", "density")


class SectionDocumentError(ValueError):
    """A section document is malformed, out of version window, or
    references something it does not define."""


class SectionDocument:
    """Declarative section description (continuum lane, ADR 0080 B1).

    Construct blank via :meth:`new`, load via :meth:`open`, mutate via
    the ``add_*`` / ``set_*`` methods (the same surface the builder
    GUI drives), persist via :meth:`save`, and realize via
    :meth:`build` — which runs a private apeGmsh session (builders →
    booleans → mesh) and returns a
    :class:`~apeGmsh.sections.SectionProperties`.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        _validate(data)
        self._data = data

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        *,
        name: str | None = None,
        kind: Literal["continuum"] = "continuum",
        units: str = "",
    ) -> "SectionDocument":
        """A blank document. ``units`` is a display label only —
        apeGmsh stays unit-agnostic."""
        return cls({
            "section_doc_version": SECTION_DOC_VERSION,
            "kind": kind,
            "name": name,
            "notes": "",
            "units": units,
            "materials": {},
            "shapes": [],
            "booleans": [],
            "mesh": {"lc": None, "order": 2},
            "disconnected": "raise",
        })

    @classmethod
    def open(cls, path: str | Path) -> "SectionDocument":
        """Load a ``.section.json`` document (version-window checked)."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SectionDocumentError(
                f"SectionDocument.open: cannot read {path!s}: {e}"
            ) from e
        return cls(data)

    def save(self, path: str | Path) -> None:
        """Write the document as deterministic, diff-friendly JSON."""
        Path(path).write_text(
            json.dumps(self._data, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    # ── introspection ────────────────────────────────────────────────

    @property
    def name(self) -> str | None:
        return self._data.get("name")

    @property
    def kind(self) -> str:
        return str(self._data["kind"])

    def to_dict(self) -> dict[str, Any]:
        """Deep copy of the underlying document dict."""
        return json.loads(json.dumps(self._data))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SectionDocument)
            and self._data == other._data
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"<SectionDocument {self.name!r}: {self.kind}, "
            f"{len(self._data['shapes'])} shape(s), "
            f"{len(self._data['materials'])} material(s)>"
        )

    # ── mutation API (what the GUI drives) ───────────────────────────

    def set_material(
        self,
        name: str,
        *,
        E: float,
        nu: float,
        G: float | None = None,
        fy: float | None = None,
        density: float | None = None,
    ) -> None:
        """Define (or redefine) a named material. Validation defers to
        :class:`SectionMaterial` at build so the rules stay single."""
        self._data["materials"][str(name)] = {
            "E": E, "nu": nu, "G": G, "fy": fy, "density": density,
        }

    def add_shape(
        self,
        shape: str,
        *,
        id: str,
        material: str | None = None,
        translate: tuple[float, float] = (0.0, 0.0),
        rotate: float | None = None,
        **params: float,
    ) -> None:
        """Add a parametric shape. ``id`` becomes the physical-group
        label; ``material`` defaults to ``id`` when materials are
        used."""
        if shape not in _SHAPE_PARAMS:
            raise SectionDocumentError(
                f"unknown shape {shape!r}; expected one of "
                f"{sorted(_SHAPE_PARAMS)} (or add_polygon)."
            )
        missing = [k for k in _SHAPE_PARAMS[shape] if k not in params]
        extra = [k for k in params if k not in _SHAPE_PARAMS[shape]]
        if missing or extra:
            raise SectionDocumentError(
                f"shape {shape!r} ({id!r}): missing params {missing}, "
                f"unknown params {extra}."
            )
        self._check_new_id(id)
        self._data["shapes"].append({
            "id": str(id), "shape": shape,
            "params": {k: float(v) for k, v in params.items()},
            "material": material,
            "translate": [float(translate[0]), float(translate[1])],
            "rotate": None if rotate is None else float(rotate),
        })

    def add_polygon(
        self,
        points: "list[tuple[float, float]]",
        *,
        id: str,
        material: str | None = None,
        translate: tuple[float, float] = (0.0, 0.0),
        rotate: float | None = None,
    ) -> None:
        """Add a freehand straight-segment polygon (the canvas tool's
        output). Points are authoring-plane vertices in order; the
        loop closes automatically."""
        if len(points) < 3:
            raise SectionDocumentError(
                f"polygon {id!r}: needs at least 3 points, "
                f"got {len(points)}."
            )
        self._check_new_id(id)
        self._data["shapes"].append({
            "id": str(id), "shape": "polygon",
            "points": [[float(x), float(y)] for x, y in points],
            "material": material,
            "translate": [float(translate[0]), float(translate[1])],
            "rotate": None if rotate is None else float(rotate),
        })

    def add_embed(self, outer: str, inner: str) -> None:
        """The composite-partition primitive: carve ``inner`` out of
        ``outer`` (cut, tool kept) then fragment the pair conformally.
        The double-cover trap is unrepresentable through this op."""
        self._check_shape_ref(outer)
        self._check_shape_ref(inner)
        self._data["booleans"].append(
            {"op": "embed", "outer": outer, "inner": inner}
        )

    def add_cut(self, target: str, tool: str, *, remove_tool: bool = True) -> None:
        """Raw boolean cut (e.g. punching holes with a sacrificial
        tool shape). For overlapping *material* regions use
        :meth:`add_embed` instead."""
        self._check_shape_ref(target)
        self._check_shape_ref(tool)
        self._data["booleans"].append({
            "op": "cut", "target": target, "tool": tool,
            "remove_tool": bool(remove_tool),
        })

    def add_fragment_pair(self, a: str, b: str) -> None:
        """Raw conformal fragment of two touching (non-overlapping)
        shapes."""
        self._check_shape_ref(a)
        self._check_shape_ref(b)
        self._data["booleans"].append({"op": "fragment_pair", "a": a, "b": b})

    def set_mesh(self, *, lc: float, order: int = 2) -> None:
        if order not in (1, 2):
            raise SectionDocumentError(f"mesh order must be 1 or 2, got {order}.")
        self._data["mesh"] = {"lc": float(lc), "order": int(order)}

    def set_disconnected(self, policy: Literal["raise", "sum"]) -> None:
        if policy not in ("raise", "sum"):
            raise SectionDocumentError(
                f"disconnected must be 'raise' or 'sum', got {policy!r}."
            )
        self._data["disconnected"] = policy

    # ── build ────────────────────────────────────────────────────────

    def build(self) -> "SectionProperties":
        """Realize the document: private apeGmsh session → builders →
        booleans → mesh → :class:`SectionProperties` (which snapshots
        the fem, so the session is closed before returning).

        Documents with an empty ``materials`` table build in the
        analyzer's geometric-only mode. Otherwise every shape's
        material (explicit or defaulted to its id) must exist in the
        table — fail-loud here, before any session is opened.
        """
        from apeGmsh import apeGmsh

        from ._analysis import SectionProperties

        data = self._data
        if data["mesh"].get("lc") is None:
            raise SectionDocumentError(
                f"{self.name or 'section document'}: set_mesh(lc=...) "
                f"before build()."
            )
        materials = self._resolve_materials()

        sacrificial = self._sacrificial_ids()

        g = apeGmsh(model_name=self.name or "section_doc", verbose=False)
        g.begin()
        try:
            instances: dict[str, Any] = {}
            for sh in data["shapes"]:
                if sh["shape"] == "polygon":
                    instances[sh["id"]] = _build_polygon(
                        g, sh, pg=sh["id"] not in sacrificial,
                    )
                else:
                    builder = getattr(g.sections, sh["shape"])
                    instances[sh["id"]] = builder(
                        **sh["params"],
                        label=sh["id"],
                        translate=tuple(sh["translate"]),
                        rotate=sh["rotate"],
                    )
            for op in data["booleans"]:
                _apply_boolean(g, op, instances)
            if sacrificial:
                g.model.geometry.remove_orphans()
            g.mesh.sizing.set_global_size(float(data["mesh"]["lc"]))
            g.mesh.generation.generate(dim=2)
            if int(data["mesh"]["order"]) > 1:
                g.mesh.generation.set_order(2)
            fem = g.mesh.queries.get_fem_data(dim=2)
        finally:
            g.end()

        return SectionProperties(
            fem,
            materials=materials or None,
            name=self.name,
            disconnected=data["disconnected"],
        )

    # ── internals ────────────────────────────────────────────────────

    def _sacrificial_ids(self) -> set[str]:
        """Shapes consumed as removed cut tools: no physical group (a
        PG a boolean empties would warn), no material requirement, and
        their consumed geometry is swept after the booleans run."""
        return {
            op["tool"]
            for op in self._data["booleans"]
            if op["op"] == "cut" and op.get("remove_tool", True)
        }

    def _resolve_materials(self) -> "dict[str, SectionMaterial]":
        table: Mapping[str, Any] = self._data["materials"]
        sacrificial = self._sacrificial_ids()
        if not table:
            for sh in self._data["shapes"]:
                if sh.get("material") is not None:
                    raise SectionDocumentError(
                        f"shape {sh['id']!r} names material "
                        f"{sh['material']!r} but the materials table is "
                        f"empty."
                    )
            return {}
        out: dict[str, SectionMaterial] = {}
        for sh in self._data["shapes"]:
            if sh["id"] in sacrificial:
                continue
            mat_name = sh.get("material") or sh["id"]
            if mat_name not in table:
                raise SectionDocumentError(
                    f"shape {sh['id']!r}: material {mat_name!r} is not "
                    f"in the materials table {sorted(table)}."
                )
            m = table[mat_name]
            out[sh["id"]] = SectionMaterial(
                E=m["E"], nu=m["nu"], G=m.get("G"),
                fy=m.get("fy"), density=m.get("density"),
            )
        return out

    def _check_new_id(self, id: str) -> None:
        if any(sh["id"] == id for sh in self._data["shapes"]):
            raise SectionDocumentError(f"duplicate shape id {id!r}.")

    def _check_shape_ref(self, id: str) -> None:
        if not any(sh["id"] == id for sh in self._data["shapes"]):
            raise SectionDocumentError(
                f"boolean references unknown shape {id!r}; defined: "
                f"{[sh['id'] for sh in self._data['shapes']]}."
            )


# ── module helpers ───────────────────────────────────────────────────


def _validate(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise SectionDocumentError("section document must be a JSON object.")
    version = data.get("section_doc_version")
    if not isinstance(version, str):
        raise SectionDocumentError(
            "missing/invalid 'section_doc_version' — not a section "
            "document."
        )
    _check_version(version)
    kind = data.get("kind")
    if kind == "fiber":
        raise SectionDocumentError(
            "fiber-lane documents are not implemented yet (ADR 0080 "
            "B2) — this build supports kind='continuum'."
        )
    if kind != "continuum":
        raise SectionDocumentError(
            f"kind must be 'continuum' (or 'fiber' once B2 ships), "
            f"got {kind!r}."
        )
    for key in ("materials", "shapes", "booleans", "mesh"):
        if key not in data:
            raise SectionDocumentError(f"missing document key {key!r}.")
    if data.get("disconnected", "raise") not in ("raise", "sum"):
        raise SectionDocumentError(
            f"disconnected must be 'raise' or 'sum', "
            f"got {data.get('disconnected')!r}."
        )
    seen: set[str] = set()
    for sh in data["shapes"]:
        sid = sh.get("id")
        if not isinstance(sid, str) or not sid:
            raise SectionDocumentError(f"shape without a string id: {sh!r}.")
        if sid in seen:
            raise SectionDocumentError(f"duplicate shape id {sid!r}.")
        seen.add(sid)
        kind_ = sh.get("shape")
        if kind_ == "polygon":
            if len(sh.get("points", ())) < 3:
                raise SectionDocumentError(
                    f"polygon {sid!r}: needs at least 3 points."
                )
        elif kind_ in _SHAPE_PARAMS:
            missing = [
                k for k in _SHAPE_PARAMS[kind_]
                if k not in sh.get("params", {})
            ]
            if missing:
                raise SectionDocumentError(
                    f"shape {sid!r} ({kind_}): missing params {missing}."
                )
        else:
            raise SectionDocumentError(
                f"shape {sid!r}: unknown shape kind {kind_!r}."
            )
    _BOOL_KEYS = {
        "embed": ("outer", "inner"),
        "cut": ("target", "tool"),
        "fragment_pair": ("a", "b"),
    }
    for op in data["booleans"]:
        kind_ = op.get("op")
        if kind_ not in _BOOL_KEYS:
            raise SectionDocumentError(
                f"unknown boolean op {kind_!r}; expected one of "
                f"{sorted(_BOOL_KEYS)}."
            )
        for key in _BOOL_KEYS[kind_]:
            ref = op.get(key)
            if ref not in seen:
                raise SectionDocumentError(
                    f"boolean {kind_!r}: {key}={ref!r} is not a defined "
                    f"shape id."
                )
    for m_name, m in dict(data["materials"]).items():
        unknown = [k for k in m if k not in _MATERIAL_KEYS]
        if unknown or "E" not in m or "nu" not in m:
            raise SectionDocumentError(
                f"material {m_name!r}: needs keys E and nu (optional "
                f"G/fy/density); unknown keys {unknown}."
            )


def _check_version(version: str) -> None:
    try:
        major, minor, _patch = (int(p) for p in version.split("."))
    except ValueError:
        raise SectionDocumentError(
            f"invalid section_doc_version {version!r}."
        ) from None
    cur_major, cur_minor, _ = (int(p) for p in SECTION_DOC_VERSION.split("."))
    if major != cur_major or not (cur_minor - 1 <= minor <= cur_minor):
        raise SectionDocumentError(
            f"section_doc_version {version} is outside this loader's "
            f"window ({cur_major}.{max(cur_minor - 1, 0)}.x – "
            f"{cur_major}.{cur_minor}.x). Upgrade apeGmsh to read a "
            f"newer document, or re-save it with a current version."
        )


def _build_polygon(g: Any, sh: dict[str, Any], *, pg: bool = True) -> Any:
    """Author one closed straight-segment polygon surface with the
    shape's id as its physical group (``pg=False`` for sacrificial
    cut tools)."""
    dx, dy = sh["translate"]
    theta = math.radians(sh["rotate"]) if sh["rotate"] is not None else None
    geo = g.model.geometry
    pts = []
    for x, y in sh["points"]:
        if theta is not None:
            x, y = (
                x * math.cos(theta) - y * math.sin(theta),
                x * math.sin(theta) + y * math.cos(theta),
            )
        pts.append(geo.add_point(x + dx, y + dy, 0.0))
    lines = [
        geo.add_line(pts[i], pts[(i + 1) % len(pts)])
        for i in range(len(pts))
    ]
    loop = geo.add_curve_loop(lines)
    surf = geo.add_plane_surface([loop])
    if pg:
        g.physical.add(2, [surf], name=sh["id"])
    return surf


def _apply_boolean(
    g: Any, op: dict[str, Any], instances: dict[str, Any]
) -> None:
    def _faces(sid: str) -> Any:
        inst = instances[sid]
        return inst.entities[2] if hasattr(inst, "entities") else inst

    kind = op["op"]
    if kind == "embed":
        g.model.boolean.cut(
            _faces(op["outer"]), _faces(op["inner"]),
            dim=2, remove_tool=False,
        )
        g.parts.fragment_pair(op["outer"], op["inner"], dim=2)
    elif kind == "cut":
        g.model.boolean.cut(
            _faces(op["target"]), _faces(op["tool"]),
            dim=2, remove_tool=bool(op.get("remove_tool", True)),
        )
    elif kind == "fragment_pair":
        g.parts.fragment_pair(op["a"], op["b"], dim=2)
    else:  # pragma: no cover - loader validates ops on mutation paths
        raise SectionDocumentError(f"unknown boolean op {kind!r}.")
