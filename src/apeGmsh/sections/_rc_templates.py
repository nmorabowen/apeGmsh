"""RC section templates — deterministic fiber-lane generators
(ADR 0080 B2).

Each template expands a small parameter set into plain fiber-lane
items (patch / layer / point dicts carrying **material role names**,
not resolved materials). Templates are stored parametric in the
:class:`~apeGmsh.sections.SectionDocument` and re-expanded on every
build, so editing ``cover`` in the GUI just works.

Conventions (documented, per "authored, never a knob"):

- Coordinates are OpenSees section-local ``(y, z)``; every template is
  centred on the origin.
- ``cover`` is measured to the **bar centre** (adjust for stirrups /
  bar radius yourself).
- ``core_split=True`` partitions the concrete exactly into a confined
  core (inside the bar-centre rectangle/ring) and a cover shell; the
  template never computes confinement parameters — you assign the
  confined/unconfined materials.
- Material roles per template: ``"concrete"`` (or ``"core"`` +
  ``"cover"`` when split) and ``"bars"``.
"""
from __future__ import annotations

import math
from typing import Any


__all__ = ["TEMPLATES", "expand_template", "template_roles"]


def _rect_bars(
    b: float, h: float, cover: float,
    bars_x: int, bars_y: int, bar_area: float,
) -> "tuple[list[dict[str, Any]], list[dict[str, Any]]]":
    """Perimeter bar layout of a rectangular section: top/bottom rows
    as straight layers (``bars_x`` bars each, corners included), side
    interiors as individual points (``bars_y − 2`` per side)."""
    zc = b / 2.0 - cover
    yc = h / 2.0 - cover
    layers = [
        {"kind": "straight", "material": "bars", "n_bars": bars_x,
         "area": bar_area, "yI": yc, "zI": -zc, "yJ": yc, "zJ": zc},
        {"kind": "straight", "material": "bars", "n_bars": bars_x,
         "area": bar_area, "yI": -yc, "zI": -zc, "yJ": -yc, "zJ": zc},
    ]
    points: list[dict[str, Any]] = []
    n_interior = bars_y - 2
    for i in range(1, n_interior + 1):
        y = -yc + 2.0 * yc * i / (n_interior + 1)
        for z in (-zc, zc):
            points.append({
                "material": "bars", "y": y, "z": z, "area": bar_area,
            })
    return layers, points


def _rc_rect(
    *, b: float, h: float, cover: float,
    bars_x: int, bars_y: int, bar_area: float,
    core_split: bool = False,
    nf_y: int = 8, nf_z: int = 8,
) -> "dict[str, list[dict[str, Any]]]":
    if cover <= 0 or 2 * cover >= min(b, h):
        raise ValueError(
            f"rc_rect_column: cover must satisfy 0 < 2*cover < "
            f"min(b, h), got cover={cover} for {b}x{h}."
        )
    if bars_x < 2 or bars_y < 2:
        raise ValueError(
            f"rc_rect_column: bars_x and bars_y count the bars per "
            f"face INCLUDING corners — need >= 2, got "
            f"bars_x={bars_x}, bars_y={bars_y}."
        )
    yc, zc = h / 2.0 - cover, b / 2.0 - cover
    if not core_split:
        patches = [{
            "kind": "rect", "material": "concrete",
            "ny": nf_y, "nz": nf_z,
            "yI": -h / 2.0, "zI": -b / 2.0, "yJ": h / 2.0, "zJ": b / 2.0,
        }]
    else:
        patches = [
            {"kind": "rect", "material": "core", "ny": nf_y, "nz": nf_z,
             "yI": -yc, "zI": -zc, "yJ": yc, "zJ": zc},
            {"kind": "rect", "material": "cover", "ny": nf_y, "nz": 2,
             "yI": -yc, "zI": -b / 2.0, "yJ": yc, "zJ": -zc},
            {"kind": "rect", "material": "cover", "ny": nf_y, "nz": 2,
             "yI": -yc, "zI": zc, "yJ": yc, "zJ": b / 2.0},
            {"kind": "rect", "material": "cover", "ny": 2, "nz": nf_z + 4,
             "yI": yc, "zI": -b / 2.0, "yJ": h / 2.0, "zJ": b / 2.0},
            {"kind": "rect", "material": "cover", "ny": 2, "nz": nf_z + 4,
             "yI": -h / 2.0, "zI": -b / 2.0, "yJ": -yc, "zJ": b / 2.0},
        ]
    layers, points = _rect_bars(b, h, cover, bars_x, bars_y, bar_area)
    return {"patches": patches, "layers": layers, "points": points}


def _rc_circ(
    *, d: float, cover: float, n_bars: int, bar_area: float,
    core_split: bool = False,
    nf_circ: int = 16, nf_rad: int = 6,
) -> "dict[str, list[dict[str, Any]]]":
    r = d / 2.0
    if cover <= 0 or cover >= r:
        raise ValueError(
            f"rc_circ_column: cover must satisfy 0 < cover < d/2, "
            f"got cover={cover} for d={d}."
        )
    if n_bars < 4:
        raise ValueError(
            f"rc_circ_column: n_bars must be >= 4, got {n_bars}."
        )
    rb = r - cover
    if not core_split:
        patches = [{
            "kind": "circ", "material": "concrete",
            "n_circ": nf_circ, "n_rad": nf_rad,
            "yC": 0.0, "zC": 0.0, "int_rad": 0.0, "ext_rad": r,
        }]
    else:
        patches = [
            {"kind": "circ", "material": "core",
             "n_circ": nf_circ, "n_rad": nf_rad,
             "yC": 0.0, "zC": 0.0, "int_rad": 0.0, "ext_rad": rb},
            {"kind": "circ", "material": "cover",
             "n_circ": nf_circ, "n_rad": 2,
             "yC": 0.0, "zC": 0.0, "int_rad": rb, "ext_rad": r},
        ]
    # bars on the ring at radius rb: first bar at +y (top), CCW
    points = []
    for i in range(n_bars):
        theta = math.pi / 2.0 + 2.0 * math.pi * i / n_bars
        points.append({
            "material": "bars",
            "y": rb * math.sin(theta),
            "z": rb * math.cos(theta),
            "area": bar_area,
        })
    return {"patches": patches, "layers": [], "points": points}


def _rc_beam(
    *, b: float, h: float, cover: float,
    top_bars: int, bottom_bars: int, bar_area: float,
    core_split: bool = False,
    nf_y: int = 10, nf_z: int = 6,
) -> "dict[str, list[dict[str, Any]]]":
    out = _rc_rect(
        b=b, h=h, cover=cover, bars_x=2, bars_y=2, bar_area=bar_area,
        core_split=core_split, nf_y=nf_y, nf_z=nf_z,
    )
    if top_bars < 2 or bottom_bars < 2:
        raise ValueError(
            f"rc_beam: top_bars and bottom_bars must be >= 2, got "
            f"{top_bars}/{bottom_bars}."
        )
    yc, zc = h / 2.0 - cover, b / 2.0 - cover
    out["layers"] = [
        {"kind": "straight", "material": "bars", "n_bars": top_bars,
         "area": bar_area, "yI": yc, "zI": -zc, "yJ": yc, "zJ": zc},
        {"kind": "straight", "material": "bars", "n_bars": bottom_bars,
         "area": bar_area, "yI": -yc, "zI": -zc, "yJ": -yc, "zJ": zc},
    ]
    out["points"] = []
    return out


#: template name → (expander, material roles without / with core_split)
TEMPLATES: "dict[str, Any]" = {
    "rc_rect_column": _rc_rect,
    "rc_circ_column": _rc_circ,
    "rc_beam": _rc_beam,
}


def template_roles(params: "dict[str, Any]") -> tuple[str, ...]:
    """Material roles a template instance requires."""
    concrete = ("core", "cover") if params.get("core_split") else ("concrete",)
    return (*concrete, "bars")


def expand_template(
    name: str, params: "dict[str, Any]",
) -> "dict[str, list[dict[str, Any]]]":
    """Expand one template deterministically into fiber-lane item
    dicts carrying material **role** names (``"concrete"`` / ``"core"``
    / ``"cover"`` / ``"bars"``)."""
    if name not in TEMPLATES:
        raise ValueError(
            f"unknown RC template {name!r}; expected one of "
            f"{sorted(TEMPLATES)}."
        )
    return TEMPLATES[name](**params)
