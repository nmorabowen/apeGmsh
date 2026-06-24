"""StructuralModel — apeGmsh-side reader for the neutral interchange document.

Parses the ``*.sm.json`` contract emitted by apeETABS (see apeETABS ADR 0009
and ``schema/structural_model.schema.json``). This is the *mirror* of the
exporter's dataclasses: the canonical spec is the JSON Schema, each repo owns
its own parse layer.

Phase 2 scope: nodes, frames, sections, materials, restraints, nodal loads.
Areas / diaphragms / distributed loads are parsed and carried through but not
yet consumed by the importer (Phases 3-4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "0.1"

Dof6 = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    x: float
    y: float
    z: float
    story: str | None = None

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class Frame:
    id: str
    i: str
    j: str
    section: str
    material: str | None = None
    kind: str | None = None
    rotation: float = 0.0


@dataclass(frozen=True, slots=True)
class Area:
    id: str
    nodes: tuple[str, ...]
    section: str
    material: str | None = None
    thickness: float | None = None
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    kind: str  # "frame" | "shell"
    material: str | None = None
    thickness: float | None = None
    props: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    E: float
    nu: float
    rho: float | None = None
    fy: float | None = None

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass(frozen=True, slots=True)
class Restraint:
    node: str
    dofs: Dof6


@dataclass(frozen=True, slots=True)
class Diaphragm:
    name: str
    nodes: tuple[str, ...]
    story: str | None = None


@dataclass(frozen=True, slots=True)
class NodalLoad:
    node: str
    force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moment_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def forces6(self) -> tuple[float, float, float, float, float, float]:
        return (*self.force_xyz, *self.moment_xyz)


@dataclass(frozen=True, slots=True)
class FrameLoad:
    frame: str
    direction: int | str
    value: float


@dataclass(frozen=True, slots=True)
class AreaLoad:
    area: str
    direction: int | str
    value: float


@dataclass(frozen=True, slots=True)
class LoadPattern:
    name: str
    nodal: tuple[NodalLoad, ...] = ()
    frame: tuple[FrameLoad, ...] = ()
    area: tuple[AreaLoad, ...] = ()


@dataclass
class StructuralModel:
    schema_version: str
    units: dict[str, str]
    nodes: list[Node]
    frames: list[Frame]
    areas: list[Area] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    restraints: list[Restraint] = field(default_factory=list)
    diaphragms: list[Diaphragm] = field(default_factory=list)
    loads: list[LoadPattern] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    # -- indexes -----------------------------------------------------------
    def node(self, nid: str) -> Node:
        return self._node_index[nid]

    def section(self, name: str) -> Section:
        return self._section_index[name]

    def material(self, name: str) -> Material:
        return self._material_index[name]

    def __post_init__(self) -> None:
        self._node_index = {n.id: n for n in self.nodes}
        self._section_index = {s.name: s for s in self.sections}
        self._material_index = {m.name: m for m in self.materials}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> "StructuralModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralModel":
        ver = d.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: document is {ver!r}, "
                f"this reader expects {SCHEMA_VERSION!r}."
            )

        def dof6(seq) -> Dof6:
            t = tuple(int(v) for v in seq)
            if len(t) != 6:
                raise ValueError(f"dof mask must have 6 entries, got {t!r}")
            return t  # type: ignore[return-value]

        loads = [
            LoadPattern(
                name=name,
                nodal=tuple(
                    NodalLoad(
                        node=l["node"],
                        force_xyz=tuple(l.get("force_xyz", (0.0, 0.0, 0.0))),
                        moment_xyz=tuple(l.get("moment_xyz", (0.0, 0.0, 0.0))),
                    )
                    for l in pat.get("nodal", [])
                ),
                frame=tuple(
                    FrameLoad(frame=l["frame"], direction=l["direction"], value=l["value"])
                    for l in pat.get("frame", [])
                ),
                area=tuple(
                    AreaLoad(area=l["area"], direction=l["direction"], value=l["value"])
                    for l in pat.get("area", [])
                ),
            )
            for name, pat in d.get("loads", {}).items()
        ]

        return cls(
            schema_version=ver,
            units=d["units"],
            source=d.get("source", {}),
            nodes=[Node(**n) for n in d["nodes"]],
            frames=[Frame(**f) for f in d["frames"]],
            areas=[
                Area(
                    id=a["id"], nodes=tuple(a["nodes"]), section=a["section"],
                    material=a.get("material"), thickness=a.get("thickness"),
                    kind=a.get("kind"),
                )
                for a in d.get("areas", [])
            ],
            sections=[
                Section(
                    name=s["name"], kind=s["kind"], material=s.get("material"),
                    thickness=s.get("thickness"), props=s.get("props", {}),
                )
                for s in d.get("sections", [])
            ],
            materials=[
                Material(
                    name=m["name"], E=m["E"], nu=m["nu"],
                    rho=m.get("rho"), fy=m.get("fy"),
                )
                for m in d.get("materials", [])
            ],
            restraints=[
                Restraint(node=r["node"], dofs=dof6(r["dofs"]))
                for r in d.get("restraints", [])
            ],
            diaphragms=[
                Diaphragm(name=dp["name"], nodes=tuple(dp["nodes"]), story=dp.get("story"))
                for dp in d.get("diaphragms", [])
            ],
            loads=loads,
        )
