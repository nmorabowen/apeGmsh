"""``SectionDocument`` — declarative section documents (ADR 0080 B1+B2).

A versioned JSON document that fully describes a section; the source
of truth for the builder GUI and the public headless API (the parity
law: every GUI action is a document mutation). Two lanes:

- **continuum** (B1): parametric shapes + freehand polygons + booleans
  + per-region materials + mesh prefs; ``build()`` runs a private
  session and returns a
  :class:`~apeGmsh.sections.SectionProperties`.
- **fiber** (B2): patches / layers / points + parametric **RC
  templates** (:mod:`._rc_templates`), expanded deterministically at
  build; ``build()`` returns a :class:`FiberRecipe` (material *names*,
  no bridge objects), and :meth:`SectionDocument.to_section` resolves
  it on an ``apeSees`` bridge — uniaxial material specs construct via
  ``ops.uniaxialMaterial.<Type>(**params)`` and the section lands as a
  registered ``ops.section.Fiber``.

Material-table entries are dual-role: continuum params (``E``/``nu``
+ optional ``G``/``fy``/``density``) and/or a ``uniaxial`` spec
(``{"type": ..., "params": {...}}``). The continuum build requires the
first role on used materials; the fiber handoff requires the second.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from ._materials import SectionMaterial
from ._rc_templates import TEMPLATES, expand_template, template_roles

if TYPE_CHECKING:  # pragma: no cover
    from ._analysis import SectionProperties


__all__ = [
    "SECTION_DOC_VERSION",
    "FiberRecipe",
    "SectionDocument",
    "SectionDocumentError",
]


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

_MATERIAL_KEYS = ("E", "nu", "G", "fy", "density", "uniaxial")


class SectionDocumentError(ValueError):
    """A section document is malformed, out of version window, or
    references something it does not define."""


def _num(value: Any, what: str) -> float:
    """A finite number (bools rejected) — the shared value gate for
    every coordinate/area/param, mutation and loader alike."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SectionDocumentError(f"{what} must be a number, got {value!r}.")
    if not math.isfinite(value):
        raise SectionDocumentError(f"{what} must be finite, got {value!r}.")
    return float(value)


def _intval(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SectionDocumentError(
            f"{what} must be an integer, got {value!r}."
        )
    if value < minimum:
        raise SectionDocumentError(
            f"{what} must be >= {minimum}, got {value}."
        )
    return value


def _pair(value: Any, what: str) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise SectionDocumentError(
            f"{what} must be a 2-vector [x, y], got {value!r}."
        )
    return (_num(value[0], f"{what}[0]"), _num(value[1], f"{what}[1]"))


def _dict_entries(value: Any, what: str) -> "list[dict[str, Any]]":
    if not isinstance(value, list) or any(
        not isinstance(e, dict) for e in value
    ):
        raise SectionDocumentError(
            f"{what} must be a list of objects, got {value!r}."
        )
    return value


def _check_bar_entry(b: dict[str, Any]) -> None:
    """Value gate for one bars-overlay entry (mutation + loader)."""
    if b.get("kind") == "point":
        _num(b.get("x"), "bar x")
        _num(b.get("y"), "bar y")
    elif b.get("kind") == "line":
        _intval(b.get("n"), "bar-line n (endpoints included)", minimum=2)
        _pair(b.get("start"), "bar-line start")
        _pair(b.get("end"), "bar-line end")
    else:
        raise SectionDocumentError(
            f"bar kind must be 'point' or 'line', got {b.get('kind')!r}."
        )
    if not isinstance(b.get("material"), str):
        raise SectionDocumentError(f"bar material must be a name: {b!r}.")
    if _num(b.get("area"), "bar area") <= 0:
        raise SectionDocumentError(
            f"bar area must be > 0, got {b.get('area')!r}."
        )


def _check_fiber_item(item: dict[str, Any], what: str) -> None:
    """Value gate for one patch/layer/point entry (mutation + loader)."""
    if not isinstance(item.get("material"), str):
        raise SectionDocumentError(
            f"{what} material must be a name: {item!r}."
        )
    if what == "rect patch":
        _intval(item.get("ny"), "rect patch ny", minimum=1)
        _intval(item.get("nz"), "rect patch nz", minimum=1)
        for k in ("yI", "zI", "yJ", "zJ"):
            _num(item.get(k), f"rect patch {k}")
    elif what == "circ patch":
        _intval(item.get("n_circ"), "circ patch n_circ", minimum=1)
        _intval(item.get("n_rad"), "circ patch n_rad", minimum=1)
        for k in ("yC", "zC", "start_ang", "end_ang"):
            if k in item:
                _num(item.get(k), f"circ patch {k}")
        ri = _num(item.get("int_rad"), "circ patch int_rad")
        re_ = _num(item.get("ext_rad"), "circ patch ext_rad")
        if not (0.0 <= ri < re_):
            raise SectionDocumentError(
                f"circ patch needs 0 <= int_rad < ext_rad, got "
                f"({ri}, {re_})."
            )
    elif what == "layer":
        _intval(item.get("n_bars"), "layer n_bars", minimum=1)
        for k in ("yI", "zI", "yJ", "zJ"):
            _num(item.get(k), f"layer {k}")
        if _num(item.get("area"), "layer area") <= 0:
            raise SectionDocumentError(
                f"layer area must be > 0, got {item.get('area')!r}."
            )
    else:  # point
        _num(item.get("y"), "point y")
        _num(item.get("z"), "point z")
        if _num(item.get("area"), "point area") <= 0:
            raise SectionDocumentError(
                f"point area must be > 0, got {item.get('area')!r}."
            )


def _expand_template_checked(
    name: Any, params: Any,
) -> "dict[str, list[dict[str, Any]]]":
    """`expand_template` with every failure wrapped as
    `SectionDocumentError` (raw TypeError/ValueError never escapes to
    document users — mutation, loader, and build all route here)."""
    if name not in TEMPLATES:
        raise SectionDocumentError(
            f"unknown RC template {name!r}; expected one of "
            f"{sorted(TEMPLATES)}."
        )
    if not isinstance(params, dict):
        raise SectionDocumentError(
            f"template {name!r}: params must be an object, got "
            f"{params!r}."
        )
    try:
        return expand_template(name, dict(params))
    except SectionDocumentError:
        raise
    except (TypeError, ValueError) as e:
        raise SectionDocumentError(
            f"template {name!r}: invalid params {sorted(params)} — {e}"
        ) from e


@dataclass(frozen=True, slots=True)
class FiberRecipe:
    """A fiber-lane document, fully expanded (ADR 0080 B2).

    Plain data — patch / layer / point dicts carrying material
    **names** from the document's table, templates already expanded.
    :meth:`SectionDocument.to_section` turns it into a registered
    bridge ``Fiber``; tests and the GUI read it directly.
    """

    patches: tuple[dict[str, Any], ...]
    layers: tuple[dict[str, Any], ...]
    points: tuple[dict[str, Any], ...]
    GJ: float | None

    def areas_by_material(self) -> dict[str, float]:
        """Total fiber area per material name (patches by geometry,
        layers/points by ``n·A``) — the exact-sum test surface."""
        out: dict[str, float] = {}

        def _add(name: str, a: float) -> None:
            out[name] = out.get(name, 0.0) + a

        for p in self.patches:
            if p["kind"] == "rect":
                _add(
                    p["material"],
                    abs((p["yJ"] - p["yI"]) * (p["zJ"] - p["zI"])),
                )
            else:
                frac = (p.get("end_ang", 360.0) - p.get("start_ang", 0.0)) / 360.0
                _add(
                    p["material"],
                    math.pi * (p["ext_rad"] ** 2 - p["int_rad"] ** 2) * frac,
                )
        for la in self.layers:
            _add(la["material"], la["n_bars"] * la["area"])
        for pt in self.points:
            _add(pt["material"], pt["area"])
        return out


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
        kind: Literal["continuum", "fiber"] = "continuum",
        units: str = "",
    ) -> "SectionDocument":
        """A blank document. ``units`` is a display label only —
        apeGmsh stays unit-agnostic."""
        data: dict[str, Any] = {
            "section_doc_version": SECTION_DOC_VERSION,
            "kind": kind,
            "name": name,
            "notes": "",
            "units": units,
            "materials": {},
        }
        if kind == "fiber":
            data |= {
                "patches": [], "layers": [], "points": [],
                "templates": [], "GJ": None,
            }
        else:
            data |= {
                "shapes": [], "booleans": [], "bars": [],
                "mesh": {"lc": None, "order": 2},
                "disconnected": "raise",
            }
        return cls(data)

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

    def export_script(self, path: "str | Path | None" = None) -> str:
        """Render the document as a readable, runnable apeGmsh script
        (ADR 0080 B4) and return the text; write it to ``path`` when
        given. One-way export — round-trip editing is the JSON
        document's job. Continuum scripts bind the analyzer to
        ``sec``; fiber scripts define ``build_section(ops)``."""
        from ._script_export import render_script

        text = render_script(self)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def save(self, path: str | Path) -> None:
        """Write the document as deterministic, diff-friendly JSON
        (strict spec - non-finite floats refuse rather than emitting
        non-portable ``NaN``/``Infinity`` tokens)."""
        try:
            text = json.dumps(
                self._data, indent=2, sort_keys=False, allow_nan=False,
            )
        except ValueError as e:
            raise SectionDocumentError(
                f"document contains non-finite numbers and cannot be "
                f"saved as portable JSON: {e}"
            ) from e
        Path(path).write_text(text + "\n", encoding="utf-8")

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
        if self.kind == "fiber":
            n_items = (
                len(self._data["patches"]) + len(self._data["layers"])
                + len(self._data["points"])
            )
            detail = (
                f"{n_items} item(s), "
                f"{len(self._data['templates'])} template(s)"
            )
        else:
            detail = f"{len(self._data['shapes'])} shape(s)"
        return (
            f"<SectionDocument {self.name!r}: {self.kind}, {detail}, "
            f"{len(self._data['materials'])} material(s)>"
        )

    # ── mutation API (what the GUI drives) ───────────────────────────

    def _require_lane(self, lane: str, method: str) -> None:
        if self.kind != lane:
            raise SectionDocumentError(
                f"{method} is a {lane}-lane operation; this document "
                f"is kind={self.kind!r}."
            )

    def set_material(
        self,
        name: str,
        *,
        E: float | None = None,
        nu: float | None = None,
        G: float | None = None,
        fy: float | None = None,
        density: float | None = None,
        uniaxial: "tuple[str, dict[str, Any]] | None" = None,
    ) -> None:
        """Define (or redefine) a named material. Dual-role: the
        continuum build needs ``E``+``nu`` on used materials; the
        fiber handoff needs ``uniaxial=("<Type>", {kwargs})`` —
        resolved as ``ops.uniaxialMaterial.<Type>(**kwargs)``. Give
        either role or both; continuum-parameter validation defers to
        :class:`SectionMaterial` at build so the rules stay single."""
        if E is None and uniaxial is None:
            raise SectionDocumentError(
                f"material {name!r}: give the continuum role "
                f"(E=, nu=) and/or the fiber role (uniaxial=)."
            )
        if (E is None) != (nu is None):
            raise SectionDocumentError(
                f"material {name!r}: E and nu come together."
            )
        entry: dict[str, Any] = {
            "E": E, "nu": nu, "G": G, "fy": fy, "density": density,
        }
        if uniaxial is not None:
            u_type, u_params = uniaxial
            entry["uniaxial"] = {
                "type": str(u_type), "params": dict(u_params),
            }
        # same value gate the loader applies (numeric, finite)
        _validate_materials({"materials": {str(name): entry}})
        self._data["materials"][str(name)] = entry

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
        self._require_lane("continuum", "add_shape")
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
        tx, ty = _pair(translate, f"shape {id!r} translate")
        self._data["shapes"].append({
            "id": str(id), "shape": shape,
            "params": {
                k: _num(v, f"shape {id!r} param {k}")
                for k, v in params.items()
            },
            "material": material,
            "translate": [tx, ty],
            "rotate": None if rotate is None
            else _num(rotate, f"shape {id!r} rotate"),
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
        self._require_lane("continuum", "add_polygon")
        if len(points) < 3:
            raise SectionDocumentError(
                f"polygon {id!r}: needs at least 3 points, "
                f"got {len(points)}."
            )
        self._check_new_id(id)
        tx, ty = _pair(translate, f"polygon {id!r} translate")
        self._data["shapes"].append({
            "id": str(id), "shape": "polygon",
            "points": [
                list(_pair(pt, f"polygon {id!r} point {i}"))
                for i, pt in enumerate(points)
            ],
            "material": material,
            "translate": [tx, ty],
            "rotate": None if rotate is None
            else _num(rotate, f"polygon {id!r} rotate"),
        })

    def add_embed(self, outer: str, inner: str) -> None:
        """The composite-partition primitive: carve ``inner`` out of
        ``outer`` (cut, tool kept) then fragment the pair conformally.
        The double-cover trap is unrepresentable through this op."""
        self._require_lane("continuum", "add_embed")
        self._check_shape_ref(outer)
        self._check_shape_ref(inner)
        if outer == inner:
            raise SectionDocumentError(
                f"embed needs two different shapes, both are {outer!r}."
            )
        self._data["booleans"].append(
            {"op": "embed", "outer": outer, "inner": inner}
        )

    def add_cut(self, target: str, tool: str, *, remove_tool: bool = True) -> None:
        """Raw boolean cut (e.g. punching holes with a sacrificial
        tool shape). For overlapping *material* regions use
        :meth:`add_embed` instead."""
        self._require_lane("continuum", "add_cut")
        self._check_shape_ref(target)
        self._check_shape_ref(tool)
        if target == tool:
            raise SectionDocumentError(
                f"cut needs two different shapes, both are {target!r}."
            )
        self._data["booleans"].append({
            "op": "cut", "target": target, "tool": tool,
            "remove_tool": bool(remove_tool),
        })

    def add_fragment_pair(self, a: str, b: str) -> None:
        """Raw conformal fragment of two touching (non-overlapping)
        shapes."""
        self._require_lane("continuum", "add_fragment_pair")
        self._check_shape_ref(a)
        self._check_shape_ref(b)
        if a == b:
            raise SectionDocumentError(
                f"fragment_pair needs two different shapes, both are "
                f"{a!r}."
            )
        self._data["booleans"].append({"op": "fragment_pair", "a": a, "b": b})

    # ── continuum bars overlay (ADR 0080 B3) ─────────────────────────

    def add_bar(
        self, *, material: str, x: float, y: float, area: float,
    ) -> None:
        """One discrete rebar on a continuum section, in **authoring
        (x, y)** coordinates. Rides the ``kind="fiber"`` lowering at
        :meth:`to_section`; concrete area is not deducted."""
        self._require_lane("continuum", "add_bar")
        entry = {
            "kind": "point", "material": str(material),
            "x": x, "y": y, "area": area,
        }
        _check_bar_entry(entry)
        self._data.setdefault("bars", []).append(entry)

    def add_bar_line(
        self,
        *,
        material: str,
        n: int,
        area: float,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        """``n`` equally spaced bars from ``start`` to ``end``
        (endpoints included, ``n >= 2``), authoring coordinates.
        Stored parametric and expanded at handoff."""
        self._require_lane("continuum", "add_bar_line")
        entry = {
            "kind": "line", "material": str(material),
            "n": n, "area": area,
            "start": list(start), "end": list(end),
        }
        _check_bar_entry(entry)
        self._data.setdefault("bars", []).append(entry)

    def _expand_bars(self) -> "list[dict[str, Any]]":
        """Bar points (authoring coords) with lines expanded."""
        out: list[dict[str, Any]] = []
        for b in self._data.get("bars", []):
            if b["kind"] == "point":
                out.append(dict(b))
            else:
                x0, y0 = b["start"]
                x1, y1 = b["end"]
                n = b["n"]
                for i in range(n):
                    t = i / (n - 1)
                    out.append({
                        "kind": "point", "material": b["material"],
                        "x": x0 + t * (x1 - x0),
                        "y": y0 + t * (y1 - y0),
                        "area": b["area"],
                    })
        return out

    # ── fiber-lane mutation (ADR 0080 B2) ────────────────────────────

    def add_patch_rect(
        self, *, material: str, ny: int, nz: int,
        yI: float, zI: float, yJ: float, zJ: float,
    ) -> None:
        self._require_lane("fiber", "add_patch_rect")
        entry = {
            "kind": "rect", "material": str(material),
            "ny": ny, "nz": nz,
            "yI": yI, "zI": zI, "yJ": yJ, "zJ": zJ,
        }
        _check_fiber_item(entry, "rect patch")
        self._data["patches"].append(entry)

    def add_patch_circ(
        self, *, material: str, n_circ: int, n_rad: int,
        ext_rad: float,
        yC: float = 0.0, zC: float = 0.0,
        int_rad: float = 0.0,
        start_ang: float = 0.0, end_ang: float = 360.0,
    ) -> None:
        self._require_lane("fiber", "add_patch_circ")
        entry = {
            "kind": "circ", "material": str(material),
            "n_circ": n_circ, "n_rad": n_rad,
            "yC": yC, "zC": zC,
            "int_rad": int_rad, "ext_rad": ext_rad,
            "start_ang": start_ang, "end_ang": end_ang,
        }
        _check_fiber_item(entry, "circ patch")
        self._data["patches"].append(entry)

    def add_layer_straight(
        self, *, material: str, n_bars: int, area: float,
        yI: float, zI: float, yJ: float, zJ: float,
    ) -> None:
        self._require_lane("fiber", "add_layer_straight")
        entry = {
            "kind": "straight", "material": str(material),
            "n_bars": n_bars, "area": area,
            "yI": yI, "zI": zI, "yJ": yJ, "zJ": zJ,
        }
        _check_fiber_item(entry, "layer")
        self._data["layers"].append(entry)

    def add_point(
        self, *, material: str, y: float, z: float, area: float,
    ) -> None:
        self._require_lane("fiber", "add_point")
        entry = {
            "material": str(material), "y": y, "z": z, "area": area,
        }
        _check_fiber_item(entry, "point")
        self._data["points"].append(entry)

    def add_template(
        self,
        template: str,
        *,
        materials: "Mapping[str, str]",
        **params: Any,
    ) -> None:
        """Add a parametric RC template (stored as parameters,
        re-expanded on every build). ``materials`` maps the template's
        roles to material-table names — exact cover required."""
        self._require_lane("fiber", "add_template")
        _expand_template_checked(template, dict(params))  # validate now
        roles = set(template_roles(dict(params)))
        given = set(materials)
        if roles != given:
            raise SectionDocumentError(
                f"template {template!r}: materials= must cover roles "
                f"{sorted(roles)} exactly — missing "
                f"{sorted(roles - given)}, unknown {sorted(given - roles)}."
            )
        self._data["templates"].append({
            "template": template,
            "params": dict(params),
            "materials": {str(k): str(v) for k, v in materials.items()},
        })

    def set_GJ(self, GJ: float | None) -> None:
        self._require_lane("fiber", "set_GJ")
        if GJ is not None and GJ <= 0:
            raise SectionDocumentError(f"GJ must be > 0, got {GJ}.")
        self._data["GJ"] = None if GJ is None else float(GJ)

    # ── continuum-lane mutation ──────────────────────────────────────

    def set_mesh(self, *, lc: float, order: int = 2) -> None:
        self._require_lane("continuum", "set_mesh")
        if order not in (1, 2):
            raise SectionDocumentError(f"mesh order must be 1 or 2, got {order}.")
        self._data["mesh"] = {"lc": float(lc), "order": int(order)}

    def set_disconnected(self, policy: Literal["raise", "sum"]) -> None:
        self._require_lane("continuum", "set_disconnected")
        if policy not in ("raise", "sum"):
            raise SectionDocumentError(
                f"disconnected must be 'raise' or 'sum', got {policy!r}."
            )
        self._data["disconnected"] = policy

    # ── build ────────────────────────────────────────────────────────

    def build(self) -> "SectionProperties | FiberRecipe":
        """Realize the document.

        Continuum lane: private apeGmsh session → builders → booleans
        → mesh → :class:`SectionProperties` (which snapshots the fem,
        so the session is closed before returning). Documents with an
        empty ``materials`` table build in the analyzer's
        geometric-only mode; otherwise every shape's material
        (explicit or defaulted to its id) must exist in the table —
        fail-loud here, before any session is opened.

        Fiber lane: templates expand deterministically and merge with
        the literal patches/layers/points into a :class:`FiberRecipe`
        (no session, no bridge objects) — hand it to
        :meth:`to_section` for the OpenSees handoff.
        """
        if self.kind == "fiber":
            return self._build_fiber()
        return self._build_continuum()

    def _build_fiber(self) -> "FiberRecipe":
        data = self._data
        table = data["materials"]
        patches = [dict(p) for p in data["patches"]]
        layers = [dict(la) for la in data["layers"]]
        points = [dict(pt) for pt in data["points"]]
        for t in data["templates"]:
            expanded = _expand_template_checked(t["template"], t["params"])
            role_map = t["materials"]
            for dst, key in (
                (patches, "patches"), (layers, "layers"),
                (points, "points"),
            ):
                for item in expanded[key]:
                    item = dict(item)
                    item["material"] = role_map[item["material"]]
                    dst.append(item)
        used = {i["material"] for i in (*patches, *layers, *points)}
        missing = sorted(u for u in used if u not in table)
        if missing:
            raise SectionDocumentError(
                f"{self.name or 'section document'}: fiber items "
                f"reference materials not in the table: {missing}."
            )
        if not (patches or layers or points):
            raise SectionDocumentError(
                f"{self.name or 'section document'}: fiber document "
                f"has no patches, layers, points, or templates."
            )
        return FiberRecipe(
            patches=tuple(patches),
            layers=tuple(layers),
            points=tuple(points),
            GJ=data.get("GJ"),
        )

    def to_section(self, ops: Any, *, name: str | None = None) -> Any:
        """Resolve the document on an ``apeSees`` bridge.

        Fiber lane: construct each used material's ``uniaxial`` spec
        via ``ops.uniaxialMaterial.<Type>(**params)`` (one bridge
        material per document material name) and register the section
        as ``ops.section.Fiber(...)``.

        Continuum lane (ADR 0080 B3): build the analyzer, resolve the
        region materials' ``uniaxial`` specs, expand the ``bars``
        overlay, and register
        ``ops.section.ComputedSection(kind="fiber", fibers=...,
        bars=...)`` — the Gauss-fiber lowering plus discrete rebar.
        (For the elastic lowering, call
        ``ops.section.ComputedSection(analysis=doc.build())``
        directly.) Fail-loud on any used material with no ``uniaxial``
        role."""
        if self.kind == "continuum":
            return self._to_computed_fiber(ops, name=name)
        from apeGmsh.opensees.section.fiber import (
            CircPatch,
            FiberPoint,
            RectPatch,
            StraightLayer,
        )

        recipe = self._build_fiber()
        used = sorted({
            i["material"]
            for i in (*recipe.patches, *recipe.layers, *recipe.points)
        })
        mats = self._resolve_uniaxial(ops, used)

        typed_patches: list[Any] = []
        for p in recipe.patches:
            if p["kind"] == "rect":
                typed_patches.append(RectPatch(
                    material=mats[p["material"]],
                    ny=p["ny"], nz=p["nz"],
                    yI=p["yI"], zI=p["zI"], yJ=p["yJ"], zJ=p["zJ"],
                ))
            else:
                typed_patches.append(CircPatch(
                    material=mats[p["material"]],
                    n_circ=p["n_circ"], n_rad=p["n_rad"],
                    yC=p["yC"], zC=p["zC"],
                    int_rad=p["int_rad"], ext_rad=p["ext_rad"],
                    start_ang=p.get("start_ang", 0.0),
                    end_ang=p.get("end_ang", 360.0),
                ))
        typed_layers = tuple(
            StraightLayer(
                material=mats[la["material"]],
                n_bars=la["n_bars"], area=la["area"],
                yI=la["yI"], zI=la["zI"], yJ=la["yJ"], zJ=la["zJ"],
            )
            for la in recipe.layers
        )
        typed_points = tuple(
            FiberPoint(
                material=mats[pt["material"]],
                y=pt["y"], z=pt["z"], area=pt["area"],
            )
            for pt in recipe.points
        )
        return ops.section.Fiber(
            patches=tuple(typed_patches),
            layers=typed_layers,
            fibers=typed_points,
            GJ=recipe.GJ,
            name=name if name is not None else self.name,
        )

    def _resolve_uniaxial(self, ops: Any, names: "list[str]") -> dict[str, Any]:
        """One bridge uniaxial material per document material name."""
        table = self._data["materials"]
        mats: dict[str, Any] = {}
        for mname in names:
            if mname not in table:
                raise SectionDocumentError(
                    f"material {mname!r} is not in the materials table "
                    f"{sorted(table)}."
                )
            spec = table[mname].get("uniaxial")
            if not spec:
                raise SectionDocumentError(
                    f"material {mname!r} has no uniaxial spec — the "
                    f"fiber handoff needs "
                    f"set_material({mname!r}, ..., uniaxial=('<Type>', "
                    f"{{...}}))."
                )
            factory = getattr(ops.uniaxialMaterial, spec["type"], None)
            if factory is None:
                raise SectionDocumentError(
                    f"material {mname!r}: ops.uniaxialMaterial has no "
                    f"constructor {spec['type']!r}."
                )
            mats[mname] = factory(**spec["params"])
        return mats

    def _to_computed_fiber(self, ops: Any, *, name: str | None) -> Any:
        from apeGmsh.opensees.section.computed import Bar

        analysis = self._build_continuum()
        sacrificial = self._sacrificial_ids()
        region_mat: dict[str, str] = {
            sh["id"]: (sh.get("material") or sh["id"])
            for sh in self._data["shapes"]
            if sh["id"] not in sacrificial
        }
        bar_points = self._expand_bars()
        used = sorted(
            set(region_mat.values())
            | {b["material"] for b in bar_points}
        )
        mats = self._resolve_uniaxial(ops, used)
        fibers = {pg: mats[mname] for pg, mname in region_mat.items()}
        bars = tuple(
            Bar(
                material=mats[b["material"]],
                x=b["x"], y=b["y"], area=b["area"],
            )
            for b in bar_points
        )
        return ops.section.ComputedSection(
            analysis=analysis,
            kind="fiber",
            fibers=fibers,
            bars=bars,
            name=name if name is not None else self.name,
        )

    def _build_continuum(self) -> "SectionProperties":
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
            if int(data["mesh"].get("order", 2)) > 1:
                g.mesh.generation.set_order(2)
            fem = g.mesh.queries.get_fem_data(dim=2)
        finally:
            g.end()

        return SectionProperties(
            fem,
            materials=materials or None,
            name=self.name,
            disconnected=data.get("disconnected", "raise"),
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
            if m.get("E") is None:
                raise SectionDocumentError(
                    f"material {mat_name!r} has no continuum role "
                    f"(E, nu) - the analyzer build needs it (the "
                    f"uniaxial role only feeds the fiber handoff)."
                )
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
    if kind not in ("continuum", "fiber"):
        raise SectionDocumentError(
            f"kind must be 'continuum' or 'fiber', got {kind!r}."
        )
    if "materials" not in data:
        raise SectionDocumentError("missing document key 'materials'.")
    _validate_materials(data)
    if kind == "fiber":
        _validate_fiber(data)
        return
    for key in ("shapes", "booleans", "mesh"):
        if key not in data:
            raise SectionDocumentError(f"missing document key {key!r}.")
    mesh = data["mesh"]
    if not isinstance(mesh, dict):
        raise SectionDocumentError(
            f"mesh must be an object {{lc, order}}, got {mesh!r}."
        )
    if mesh.get("lc") is not None:
        if _num(mesh["lc"], "mesh lc") <= 0:
            raise SectionDocumentError(
                f"mesh lc must be > 0, got {mesh['lc']!r}."
            )
    if mesh.get("order", 2) not in (1, 2):
        raise SectionDocumentError(
            f"mesh order must be 1 or 2, got {mesh.get('order')!r}."
        )
    # "bars" is a 1.0-additive key (B3) — absent in earlier documents
    for b in _dict_entries(data.get("bars", []), "bars"):
        _check_bar_entry(b)
    if data.get("disconnected", "raise") not in ("raise", "sum"):
        raise SectionDocumentError(
            f"disconnected must be 'raise' or 'sum', "
            f"got {data.get('disconnected')!r}."
        )
    seen: set[str] = set()
    for sh in _dict_entries(data["shapes"], "shapes"):
        sid = sh.get("id")
        if not isinstance(sid, str) or not sid:
            raise SectionDocumentError(f"shape without a string id: {sh!r}.")
        if sid in seen:
            raise SectionDocumentError(f"duplicate shape id {sid!r}.")
        seen.add(sid)
        _pair(sh.get("translate", (0.0, 0.0)), f"shape {sid!r} translate")
        if sh.get("rotate") is not None:
            _num(sh["rotate"], f"shape {sid!r} rotate")
        kind_ = sh.get("shape")
        if kind_ == "polygon":
            pts = sh.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                raise SectionDocumentError(
                    f"polygon {sid!r}: needs a list of at least 3 points."
                )
            for i, pt in enumerate(pts):
                _pair(pt, f"polygon {sid!r} point {i}")
        elif kind_ in _SHAPE_PARAMS:
            params = sh.get("params")
            if not isinstance(params, dict):
                raise SectionDocumentError(
                    f"shape {sid!r} ({kind_}): params must be an object."
                )
            missing = [k for k in _SHAPE_PARAMS[kind_] if k not in params]
            if missing:
                raise SectionDocumentError(
                    f"shape {sid!r} ({kind_}): missing params {missing}."
                )
            for k in _SHAPE_PARAMS[kind_]:
                _num(params[k], f"shape {sid!r} param {k}")
        else:
            raise SectionDocumentError(
                f"shape {sid!r}: unknown shape kind {kind_!r}."
            )
    _BOOL_KEYS = {
        "embed": ("outer", "inner"),
        "cut": ("target", "tool"),
        "fragment_pair": ("a", "b"),
    }
    for op in _dict_entries(data["booleans"], "booleans"):
        kind_ = op.get("op")
        if kind_ not in _BOOL_KEYS:
            raise SectionDocumentError(
                f"unknown boolean op {kind_!r}; expected one of "
                f"{sorted(_BOOL_KEYS)}."
            )
        key_a, key_b = _BOOL_KEYS[kind_]
        for key in (key_a, key_b):
            ref = op.get(key)
            if ref not in seen:
                raise SectionDocumentError(
                    f"boolean {kind_!r}: {key}={ref!r} is not a defined "
                    f"shape id."
                )
        if op.get(key_a) == op.get(key_b):
            raise SectionDocumentError(
                f"boolean {kind_!r}: {key_a} and {key_b} must be "
                f"different shapes, both are {op.get(key_a)!r}."
            )


def _validate_materials(data: dict[str, Any]) -> None:
    table = data["materials"]
    if not isinstance(table, dict):
        raise SectionDocumentError(
            f"materials must be an object of named entries, got "
            f"{table!r}."
        )
    for m_name, m in table.items():
        if not isinstance(m, dict):
            raise SectionDocumentError(
                f"material {m_name!r}: entry must be an object, got "
                f"{m!r}."
            )
        unknown = [k for k in m if k not in _MATERIAL_KEYS]
        has_continuum = m.get("E") is not None
        has_uniaxial = m.get("uniaxial") is not None
        if unknown or not (has_continuum or has_uniaxial):
            raise SectionDocumentError(
                f"material {m_name!r}: needs the continuum role (E, "
                f"nu) and/or a uniaxial spec; unknown keys {unknown}."
            )
        if has_continuum:
            if m.get("nu") is None:
                raise SectionDocumentError(
                    f"material {m_name!r}: E and nu come together."
                )
            _num(m["E"], f"material {m_name!r} E")
            _num(m["nu"], f"material {m_name!r} nu")
            for k in ("G", "fy", "density"):
                if m.get(k) is not None:
                    _num(m[k], f"material {m_name!r} {k}")
        if has_uniaxial:
            u = m["uniaxial"]
            if (
                not isinstance(u, dict)
                or not isinstance(u.get("type"), str)
                or not isinstance(u.get("params"), dict)
            ):
                raise SectionDocumentError(
                    f"material {m_name!r}: uniaxial spec must be "
                    f"{{'type': str, 'params': dict}}, got {u!r}."
                )


_PATCH_RECT_KEYS = ("material", "ny", "nz", "yI", "zI", "yJ", "zJ")
_PATCH_CIRC_KEYS = (
    "material", "n_circ", "n_rad", "yC", "zC", "int_rad", "ext_rad",
)
_LAYER_KEYS = ("material", "n_bars", "area", "yI", "zI", "yJ", "zJ")
_POINT_KEYS = ("material", "y", "z", "area")


def _validate_fiber(data: dict[str, Any]) -> None:
    for key in ("patches", "layers", "points", "templates"):
        if key not in data:
            raise SectionDocumentError(f"missing document key {key!r}.")
    gj = data.get("GJ")
    if gj is not None and (
        isinstance(gj, bool)
        or not isinstance(gj, (int, float))
        or not math.isfinite(gj)
        or gj <= 0
    ):
        raise SectionDocumentError(f"GJ must be > 0 or null, got {gj!r}.")

    def _need(item: dict[str, Any], keys: tuple[str, ...], what: str) -> None:
        missing = [k for k in keys if k not in item]
        if missing:
            raise SectionDocumentError(
                f"{what} entry missing keys {missing}: {item!r}."
            )
        _check_fiber_item(item, what)

    for p in _dict_entries(data["patches"], "patches"):
        if p.get("kind") == "rect":
            _need(p, _PATCH_RECT_KEYS, "rect patch")
        elif p.get("kind") == "circ":
            _need(p, _PATCH_CIRC_KEYS, "circ patch")
        else:
            raise SectionDocumentError(
                f"patch kind must be 'rect' or 'circ', got "
                f"{p.get('kind')!r}."
            )
    for la in _dict_entries(data["layers"], "layers"):
        _need(la, _LAYER_KEYS, "layer")
    for pt in _dict_entries(data["points"], "points"):
        _need(pt, _POINT_KEYS, "point")
    from ._rc_templates import template_roles as _roles

    for t in _dict_entries(data["templates"], "templates"):
        tname = t.get("template")
        role_map = t.get("materials")
        # validates the name, the params dict, and the params VALUES
        # (cover gates, unknown/non-numeric params) at load time
        params = t.get("params")
        _expand_template_checked(tname, params)
        if not isinstance(role_map, dict):
            raise SectionDocumentError(
                f"template {tname!r}: needs a 'materials' role map."
            )
        roles = set(_roles(params))
        if roles != set(role_map):
            raise SectionDocumentError(
                f"template {tname!r}: materials= must cover roles "
                f"{sorted(roles)} exactly, got {sorted(role_map)}."
            )


def _check_version(version: str) -> None:
    import re

    # canonical digits only — rejects negatives ("1.-1.0" would slip
    # the minor-window arithmetic at minor 0), whitespace, and
    # zero-padding
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise SectionDocumentError(
            f"invalid section_doc_version {version!r}."
        )
    major, minor, _patch = (int(p) for p in version.split("."))
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
