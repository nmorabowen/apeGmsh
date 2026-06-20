"""
Stage 1 — Reinforcement-cage Definitions (pre-mesh, geometry-level intent).

These dataclasses describe *what* reinforcement a user wants — bars,
stirrups, hooks, a whole cage — at the geometry level, as **pure,
frozen, serialisable data**. They carry no node tags, no OpenSees
handles, and no live gmsh entities (Option-B layering, mirroring
:class:`~apeGmsh._kernel.defs.constraints.ReinforceDef`). The L2
``g.rebar`` composite (:class:`~apeGmsh.core.RebarComposite`) generates
geometry from them and routes coupling; the L1 spec itself stays inert
and round-trips through ``to_dict`` / ``from_dict``.

Lengths are **unitless** here — a bare ``float`` is in model units, and a
``"<k>db"`` string (e.g. ``"12db"``) is resolved against the bar
diameter by a :class:`DetailingStandard` at *bind* time, never at
construction. This keeps the spec serialisable and standard-agnostic.

See ADR 0066 (reinforcement-cage authoring) for the design rationale and
the L1/L2/L3 split. The detailing standards + bar catalogue live in
:mod:`apeGmsh.rebar.detailing`; geometry generation + coupling live in
:mod:`apeGmsh.core.RebarComposite`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

Vec3 = tuple[float, float, float]

# A "<k>db" length token: a positive number immediately followed by "db",
# e.g. "12db", "4db", "2.5db". Resolved to k * bar_diameter at bind time.
_DB_TOKEN = re.compile(r"^\s*\d+(?:\.\d+)?\s*db\s*$", re.IGNORECASE)

# Sentinel for "store the bend/corner radius as metadata, emit a sharp
# polyline corner" — the default fidelity (ADR 0066 §5).
METADATA = "metadata"

# Bend-plane selector tokens accepted by Hook.turn (besides an explicit
# 3-vector). Resolution happens at bind time in the L2 composite.
_TURN_TOKENS = frozenset(
    {"centroid", "in", "out", "up", "down",
     "+x", "-x", "+y", "-y", "+z", "-z"}
)


def _is_db_token(value: Any) -> bool:
    """True iff *value* is a ``"<k>db"`` length string."""
    return isinstance(value, str) and bool(_DB_TOKEN.match(value))


def _validate_length(value: float | str, *, field_name: str, owner: str,
                     allow_metadata: bool = False) -> None:
    """Fail-loud check for a length field: a positive float, a ``"<k>db"``
    token, or (optionally) the ``"metadata"`` sentinel."""
    if allow_metadata and value == METADATA:
        return
    if isinstance(value, str):
        if _is_db_token(value):
            return
        extra = ' or "metadata"' if allow_metadata else ""
        raise ValueError(
            f'{owner}: {field_name}={value!r} is not a valid length. '
            f'Use a positive number (model units) or a "<k>db" string '
            f'such as "12db"{extra}.'
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{owner}: {field_name} must be a number or a \"<k>db\" "
            f"string, got {type(value).__name__}."
        )
    if value <= 0.0:
        raise ValueError(
            f"{owner}: {field_name} must be > 0, got {value}."
        )


def _validate_db(value: float | str, *, owner: str) -> None:
    """A bar size: a positive float (model-unit diameter) or a catalogue
    designation string (e.g. ``"#8"``, ``"20mm"``, ``"20M"``). The
    designation is resolved by a :class:`DetailingStandard` at bind time,
    so here we only reject the empty/obviously-wrong cases."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{owner}: db designation is empty.")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{owner}: db must be a positive number or a designation "
            f"string (e.g. \"#8\"), got {type(value).__name__}."
        )
    if value <= 0.0:
        raise ValueError(f"{owner}: db must be > 0, got {value}.")


# ── Hook ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hook:
    """A bend detail at a **free end** of a bar/stirrup path (ADR 0066 §3).

    The incoming tangent comes from the parent path — it is *not* stored
    here. The bend *plane* is fixed at bind time from :attr:`turn`
    (default ``"centroid"`` ⇒ toward the host/cage centroid).

    ``bend_radius=None`` defers to the :class:`DetailingStandard`
    (``min_bend_diameter``); ``tail`` / ``bend_radius`` accept a number or
    a ``"<k>db"`` token. ``true_arc=False`` (default) emits a sharp
    polyline corner with the radius carried as metadata; ``true_arc=True``
    builds a real fillet (L2 concern).
    """

    angle: float                                  # 90 | 135 | 180 (deg)
    tail: float | str                             # length or "<k>db"
    bend_radius: float | str | None = None        # None ⇒ from standard
    turn: str | Vec3 = "centroid"                 # bend-plane selector
    true_arc: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.angle, (int, float)) or isinstance(self.angle, bool):
            raise ValueError(f"Hook: angle must be a number, got {self.angle!r}.")
        if not (0.0 < float(self.angle) <= 180.0):
            raise ValueError(
                f"Hook: angle must be in (0, 180] degrees, got {self.angle}."
            )
        _validate_length(self.tail, field_name="tail", owner="Hook")
        if self.bend_radius is not None:
            _validate_length(self.bend_radius, field_name="bend_radius",
                             owner="Hook")
        self._validate_turn()

    def _validate_turn(self) -> None:
        turn = self.turn
        if isinstance(turn, str):
            if turn.lower() not in _TURN_TOKENS:
                raise ValueError(
                    f"Hook: turn={turn!r} is not a known token "
                    f"({sorted(_TURN_TOKENS)}) or a 3-vector."
                )
            return
        if (isinstance(turn, tuple) and len(turn) == 3
                and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                        for c in turn)):
            if all(c == 0.0 for c in turn):
                raise ValueError("Hook: turn vector must be non-zero.")
            return
        raise ValueError(
            f"Hook: turn must be a token string or an (x, y, z) vector, "
            f"got {turn!r}."
        )

    # Code-aware factories. They set only the ACI-standard angle + a
    # "<k>db" tail default and leave bend_radius=None so the
    # DetailingStandard fills it at bind time (no standard needed here).
    @classmethod
    def seismic_135(cls, *, tail: float | str = "6db",
                    turn: str | Vec3 = "centroid", true_arc: bool = False,
                    name: str | None = None) -> "Hook":
        return cls(angle=135.0, tail=tail, turn=turn, true_arc=true_arc,
                   name=name)

    @classmethod
    def standard_90(cls, *, tail: float | str = "12db",
                    turn: str | Vec3 = "centroid", true_arc: bool = False,
                    name: str | None = None) -> "Hook":
        return cls(angle=90.0, tail=tail, turn=turn, true_arc=true_arc,
                   name=name)

    @classmethod
    def standard_180(cls, *, tail: float | str = "4db",
                     turn: str | Vec3 = "centroid", true_arc: bool = False,
                     name: str | None = None) -> "Hook":
        return cls(angle=180.0, tail=tail, turn=turn, true_arc=true_arc,
                   name=name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle": float(self.angle),
            "tail": self.tail,
            "bend_radius": self.bend_radius,
            "turn": list(self.turn) if isinstance(self.turn, tuple) else self.turn,
            "true_arc": bool(self.true_arc),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hook":
        turn = d.get("turn", "centroid")
        if isinstance(turn, list):
            turn = tuple(turn)
        return cls(
            angle=d["angle"], tail=d["tail"], bend_radius=d.get("bend_radius"),
            turn=turn, true_arc=d.get("true_arc", False), name=d.get("name"),
        )


# ── Path ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Path:
    """An ordered open polyline of 3D control points, with a bend radius
    applied at every interior vertex (ADR 0066 §3).

    ``corner_radius="metadata"`` (default) emits sharp polyline corners
    and stores the radius as metadata; a number or ``"<k>db"`` token sets
    a true fillet radius consumed by the L2 builder under ``true_arc``.
    """

    points: tuple[Vec3, ...]
    corner_radius: float | str = METADATA

    def __post_init__(self) -> None:
        pts = tuple(tuple(float(c) for c in p) for p in self.points)
        for p in pts:
            if len(p) != 3:
                raise ValueError(
                    f"Path: every point must be (x, y, z), got {p!r}."
                )
        if len(pts) < 2:
            raise ValueError(
                f"Path: need at least 2 points, got {len(pts)}."
            )
        object.__setattr__(self, "points", pts)
        _validate_length(self.corner_radius, field_name="corner_radius",
                         owner="Path", allow_metadata=True)

    @property
    def is_metadata_bend(self) -> bool:
        return self.corner_radius == METADATA

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [list(p) for p in self.points],
            "corner_radius": self.corner_radius,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Path":
        return cls(
            points=tuple(tuple(p) for p in d["points"]),
            corner_radius=d.get("corner_radius", METADATA),
        )


# ── Bar ──────────────────────────────────────────────────────────────

_ELEMENT_KINDS = frozenset({"truss", "beam"})


@dataclass(frozen=True)
class Bar:
    """A single reinforcing bar: a :class:`Path` centerline + diameter +
    material (by name) + optional end hooks (ADR 0066 §3, §7).

    ``element="truss"`` (default, realised as ``CorotTruss``) is
    axial-only; ``element="beam"`` is opt-in dowel action and is gated on
    the ADR-0010 Phase-4 orientation fan-out for curved bars (enforced at
    bridge time, not here).
    """

    path: Path
    db: float | str
    material: str
    role: str = "longitudinal"
    element: str = "truss"
    start_hook: Hook | None = None
    end_hook: Hook | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError(
                f"Bar: path must be a Path, got {type(self.path).__name__}."
            )
        _validate_db(self.db, owner="Bar")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Bar: material must be a non-empty name string.")
        if self.element not in _ELEMENT_KINDS:
            raise ValueError(
                f"Bar: element must be one of {sorted(_ELEMENT_KINDS)}, "
                f"got {self.element!r}."
            )
        for h, slot in ((self.start_hook, "start_hook"),
                        (self.end_hook, "end_hook")):
            if h is not None and not isinstance(h, Hook):
                raise ValueError(
                    f"Bar: {slot} must be a Hook or None, "
                    f"got {type(h).__name__}."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "db": self.db,
            "material": self.material,
            "role": self.role,
            "element": self.element,
            "start_hook": self.start_hook.to_dict() if self.start_hook else None,
            "end_hook": self.end_hook.to_dict() if self.end_hook else None,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Bar":
        sh, eh = d.get("start_hook"), d.get("end_hook")
        return cls(
            path=Path.from_dict(d["path"]),
            db=d["db"], material=d["material"],
            role=d.get("role", "longitudinal"),
            element=d.get("element", "truss"),
            start_hook=Hook.from_dict(sh) if sh else None,
            end_hook=Hook.from_dict(eh) if eh else None,
            name=d.get("name"),
        )


# ── Stirrup ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stirrup:
    """A tie / hoop: an **open** corner :class:`Path` that returns near
    its origin with a hooked closure (ADR 0066 §3). A real tie is *not* a
    closed loop — the two hooked tails overlap at one corner; the L2
    builder realises the closure-overlap geometry.
    """

    path: Path
    db: float | str
    material: str
    role: str = "tie"
    closure_hook: Hook = field(default_factory=Hook.seismic_135)
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError(
                f"Stirrup: path must be a Path, got {type(self.path).__name__}."
            )
        _validate_db(self.db, owner="Stirrup")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Stirrup: material must be a non-empty name string.")
        if not isinstance(self.closure_hook, Hook):
            raise ValueError(
                f"Stirrup: closure_hook must be a Hook, "
                f"got {type(self.closure_hook).__name__}."
            )

    @classmethod
    def rect(cls, bx: float, by: float, cover: float, *, db: float | str,
             material: str, z: float = 0.0, db_value: float | None = None,
             closure_hook: Hook | None = None, role: str = "tie",
             name: str | None = None) -> "Stirrup":
        """A rectangular tie at height ``z``. ``cover`` is clear cover to
        the *outside* of the tie bar; the centerline is inset by
        ``cover + db/2``. ``db_value`` (model-unit diameter) is used for
        the inset when ``db`` is a designation string; for a numeric
        ``db`` it defaults to ``db``.
        """
        d = db_value if db_value is not None else (db if isinstance(db, (int, float)) else None)
        if d is None:
            raise ValueError(
                "Stirrup.rect: pass db_value=<diameter> when db is a "
                "designation string, so the cover→centerline inset is known."
            )
        inset = cover + d / 2.0
        if 2.0 * inset >= min(bx, by):
            raise ValueError(
                f"Stirrup.rect: cover+db/2={inset} too large for section "
                f"{bx}x{by} (degenerate tie polygon)."
            )
        x0, y0 = inset, inset
        x1, y1 = bx - inset, by - inset
        # open corner polyline that returns to the start corner (closure
        # overlap + seam stagger is an L2 geometry concern)
        corners = [
            (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z),
        ]
        return cls(
            path=Path(points=tuple(corners)),
            db=db, material=material, role=role,
            closure_hook=closure_hook or Hook.seismic_135(), name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "db": self.db,
            "material": self.material,
            "role": self.role,
            "closure_hook": self.closure_hook.to_dict(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Stirrup":
        return cls(
            path=Path.from_dict(d["path"]),
            db=d["db"], material=d["material"], role=d.get("role", "tie"),
            closure_hook=Hook.from_dict(d["closure_hook"]),
            name=d.get("name"),
        )


# ── Cage ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cage:
    """A reinforcement cage: bars + stirrups, the serialisable source of
    truth that feeds both binding surfaces (inline + composed-Part).

    The optional ``standard`` is a :class:`DetailingStandard` reference;
    it is *not* serialised by name here (the L2 composite re-binds it at
    ``place`` time), so ``to_dict`` carries only the geometry+intent.
    """

    bars: tuple[Bar, ...] = ()
    stirrups: tuple[Stirrup, ...] = ()
    standard: Any = None              # DetailingStandard | None (not serialised)

    def __post_init__(self) -> None:
        bars = tuple(self.bars)
        stirrups = tuple(self.stirrups)
        for b in bars:
            if not isinstance(b, Bar):
                raise ValueError(
                    f"Cage: bars must contain Bar instances, "
                    f"got {type(b).__name__}."
                )
        for s in stirrups:
            if not isinstance(s, Stirrup):
                raise ValueError(
                    f"Cage: stirrups must contain Stirrup instances, "
                    f"got {type(s).__name__}."
                )
        if not bars and not stirrups:
            raise ValueError("Cage: empty — needs at least one bar or stirrup.")
        object.__setattr__(self, "bars", bars)
        object.__setattr__(self, "stirrups", stirrups)

    def to_dict(self) -> dict[str, Any]:
        """Geometry+intent only (no standard, no OpenSees handles)."""
        return {
            "bars": [b.to_dict() for b in self.bars],
            "stirrups": [s.to_dict() for s in self.stirrups],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cage":
        return cls(
            bars=tuple(Bar.from_dict(b) for b in d.get("bars", [])),
            stirrups=tuple(Stirrup.from_dict(s) for s in d.get("stirrups", [])),
        )


__all__ = [
    "Vec3", "METADATA", "Hook", "Path", "Bar", "Stirrup", "Cage",
]
