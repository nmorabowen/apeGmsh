"""Names-first ``SelectionEnvelope`` projector (ADR 0095 S1).

Pure: no ``gmsh``, no Qt, no ``SelectionState`` import. Callers pass
picked targets (``SelectionTarget`` or ``(dim, tag)``) plus a name
index. Identity fields are labels / physical groups / phase /
substrate; dimtags and node ids are evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Union,
)

Phase = Literal["model", "mesh", "results"]
ENVELOPE_SCHEMA = 1

NamesLookup = Union[
    Mapping[tuple[int, int], "NameRecord"],
    Callable[[int, int], "NameRecord"],
]


@dataclass(frozen=True)
class NameRecord:
    """Names (and optional bbox) for one picked entity."""

    labels: tuple[str, ...] = ()
    physical_groups: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class PickEvidence:
    """One pick as evidence — tags / ids, plus any names already known."""

    substrate: str
    dim: int
    key: int
    labels: tuple[str, ...] = ()
    physical_groups: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class SelectionEnvelope:
    """JSON payload the host writes on pick change (ADR 0095 Part 3).

    Identity: ``labels``, ``physical_groups``, ``phase``, ``substrate``,
    ``bbox``, ``unnamed``. ``evidence`` carries dimtags / node ids.
    """

    phase: Phase
    substrate: str
    labels: tuple[str, ...]
    physical_groups: tuple[str, ...]
    unnamed: tuple[PickEvidence, ...]
    evidence: tuple[PickEvidence, ...]
    bbox: tuple[float, float, float, float, float, float] | None = None
    active_group: str | None = None
    schema: int = ENVELOPE_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


def _as_pick(target: object) -> tuple[str, int, int]:
    """Normalise a ``SelectionTarget`` or ``(dim, tag)`` to substrate/dim/key."""
    if isinstance(target, tuple) and len(target) == 2:
        return "model_brep", int(target[0]), int(target[1])
    substrate = getattr(target, "substrate", None)
    if substrate is None:
        raise TypeError(
            f"pick target must be SelectionTarget or (dim, tag); "
            f"got {type(target).__name__}"
        )
    value = substrate.value if hasattr(substrate, "value") else str(substrate)
    return str(value), int(target.dim), int(target.key)


def _lookup(names: NamesLookup | None, dim: int, key: int) -> NameRecord:
    if names is None:
        return NameRecord()
    if callable(names):
        rec = names(dim, key)
        return rec if rec is not None else NameRecord()
    return names.get((dim, key), NameRecord())


def _unique(seq: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _union_bbox(
    boxes: Sequence[tuple[float, float, float, float, float, float] | None],
) -> tuple[float, float, float, float, float, float] | None:
    present = [b for b in boxes if b is not None]
    if not present:
        return None
    xmin = min(b[0] for b in present)
    ymin = min(b[1] for b in present)
    zmin = min(b[2] for b in present)
    xmax = max(b[3] for b in present)
    ymax = max(b[4] for b in present)
    zmax = max(b[5] for b in present)
    return (xmin, ymin, zmin, xmax, ymax, zmax)


def project(
    targets: Sequence[object],
    *,
    phase: Phase,
    names: NamesLookup | None = None,
    active_group: str | None = None,
) -> SelectionEnvelope:
    """Project picked targets + a name index into a :class:`SelectionEnvelope`."""
    evidence: list[PickEvidence] = []
    substrates: list[str] = []
    for target in targets:
        substrate, dim, key = _as_pick(target)
        rec = _lookup(names, dim, key)
        evidence.append(
            PickEvidence(
                substrate=substrate,
                dim=dim,
                key=key,
                labels=tuple(rec.labels),
                physical_groups=tuple(rec.physical_groups),
                bbox=rec.bbox,
            )
        )
        substrates.append(substrate)

    unique_sub = _unique(substrates)
    if not unique_sub:
        substrate = {
            "model": "model_brep",
            "mesh": "mesh_topo",
            "results": "results_topo",
        }[phase]
    elif len(unique_sub) == 1:
        substrate = unique_sub[0]
    else:
        substrate = "mixed"

    unnamed = tuple(
        e for e in evidence if not e.labels and not e.physical_groups
    )
    return SelectionEnvelope(
        phase=phase,
        substrate=substrate,
        labels=_unique(name for e in evidence for name in e.labels),
        physical_groups=_unique(
            name for e in evidence for name in e.physical_groups
        ),
        unnamed=unnamed,
        evidence=tuple(evidence),
        bbox=_union_bbox([e.bbox for e in evidence]),
        active_group=active_group,
    )


def project_state(
    state: object,
    *,
    phase: Phase,
    names: NamesLookup | None = None,
) -> SelectionEnvelope:
    """Project a ``SelectionState`` (duck-typed) into an envelope."""
    targets = getattr(state, "targets", None)
    if targets is None:
        targets = getattr(state, "picks", ())
    return project(
        list(targets),
        phase=phase,
        names=names,
        active_group=getattr(state, "active_group", None),
    )


def dumps(envelope: SelectionEnvelope) -> str:
    return json.dumps(envelope.to_dict(), indent=2) + "\n"


def _evidence_from_dict(raw: object) -> PickEvidence:
    if not isinstance(raw, dict):
        raise ValueError("evidence item must be an object")
    bbox = raw.get("bbox")
    if bbox is not None:
        bbox = tuple(float(x) for x in bbox)
        if len(bbox) != 6:
            raise ValueError("bbox must be six floats")
    return PickEvidence(
        substrate=str(raw["substrate"]),
        dim=int(raw["dim"]),
        key=int(raw["key"]),
        labels=tuple(raw.get("labels") or ()),
        physical_groups=tuple(raw.get("physical_groups") or ()),
        bbox=bbox,
    )


def loads(text: str) -> SelectionEnvelope:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("envelope JSON must be an object")
    bbox = data.get("bbox")
    if bbox is not None:
        bbox = tuple(float(x) for x in bbox)
        if len(bbox) != 6:
            raise ValueError("bbox must be six floats")
    phase = data.get("phase")
    if phase not in ("model", "mesh", "results"):
        raise ValueError(f"unknown phase {phase!r}")
    return SelectionEnvelope(
        phase=phase,
        substrate=str(data.get("substrate") or "model_brep"),
        labels=tuple(data.get("labels") or ()),
        physical_groups=tuple(data.get("physical_groups") or ()),
        unnamed=tuple(_evidence_from_dict(x) for x in (data.get("unnamed") or ())),
        evidence=tuple(
            _evidence_from_dict(x) for x in (data.get("evidence") or ())
        ),
        bbox=bbox,
        active_group=data.get("active_group"),
        schema=int(data.get("schema") or ENVELOPE_SCHEMA),
    )


def write_envelope(path: Path, envelope: SelectionEnvelope) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(envelope), encoding="utf-8")
    return path


def read_envelope(path: Path) -> SelectionEnvelope:
    return loads(Path(path).read_text(encoding="utf-8"))
