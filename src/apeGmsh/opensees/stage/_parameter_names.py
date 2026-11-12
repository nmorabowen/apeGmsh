"""The ``setParameter`` names of the element and material classes that
``s.update_parameter`` targets, and the check that reads them (#2377).

``s.update_parameter(name, value, pg=...)`` emits ``addToParameter $pid
element $tag <name> [<mat_tag>]``.  An element whose ``setParameter``
does not identify ``<name>`` makes OpenSees print ``Parameter::addComponent
<pid> -- no objects were able to identify parameter`` and carry on, so the
update is lost while the user believes it ran.  The check refuses that by
element class, under the owner ruling of 2026-11-11, "(b) + (c)":

* An element class in :data:`ELEMENT_PARAMETER_NAMES` that neither takes
  the name nor forwards unmatched names to its materials is refused.
* A class that forwards (``FourNodeQuad``, ``Truss``, ``SSPquadUP``,
  ``LadrunoUP``) is refused only when every material it carries is in
  :data:`MATERIAL_PARAMETER_NAMES` and none of them takes the name.
* An element or material class that is not catalogued warns once per class
  (:class:`UncataloguedParameterWarning`) and the update emits, so a
  parameter that works is never blocked.

Every entry cites the C++ ``setParameter`` it was read from (fork checkout,
``ladruno`` f37e2ff19; the bodies are upstream's unless the class is a
fork class).  A name the element matches with ``strstr`` (any name that
*contains* ``"material"`` or ``"damp"``) is claimed by the element, so it
is never forwarded; such a name is not decided here.  The stage-flag names
(``materialState``, ``updateMaterialStage``) have their own route checks
in ``stage/_builder.py`` and are not checked here.
"""

from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass
from typing import Iterable, Sequence

from .._internal.build.errors import BridgeError
from ..material.nd_soil import SET_PARAMETER_NAMES as _SOIL_NAMES

__all__ = [
    "ELEMENT_PARAMETER_NAMES",
    "ELEMENT_PARAMETER_NAMES_BY_NDM",
    "MATERIAL_PARAMETER_NAMES",
    "ElementNames",
    "MaterialNames",
    "ParameterTarget",
    "UncataloguedParameterWarning",
    "check_parameter_name",
    "element_names",
]


class UncataloguedParameterWarning(UserWarning):
    """``s.update_parameter`` targets an element or material class whose
    ``setParameter`` names are not catalogued, so the bridge cannot tell
    whether the update reaches anything.  The update emits."""


@dataclass(frozen=True)
class ElementNames:
    """What an element class's ``setParameter`` does with a name.

    ``owns`` are the names it matches with ``strcmp``.  ``claims`` are the
    substrings it matches with ``strstr``: a name containing one is taken
    by the element and never forwarded.  ``forwards`` is whether it hands
    every other name to the materials it carries.  ``source`` is the
    ``path::Class::setParameter`` it was read from.
    """

    owns: frozenset[str]
    claims: frozenset[str]
    forwards: bool
    source: str

    def takes(self, name: str) -> bool:
        return name in self.owns or any(s in name for s in self.claims)


@dataclass(frozen=True)
class MaterialNames:
    """What a material class's ``setParameter`` matches.

    ``tag_checked`` classes match only when ``argv[1]`` is their own tag,
    which ``s.update_parameter(..., material=m)`` appends; without it, or
    with another material's tag, they take nothing.
    """

    owns: frozenset[str]
    tag_checked: bool
    source: str


_FOURNODEQUAD = "SRC/element/fourNodeQuad/FourNodeQuad.cpp::FourNodeQuad::setParameter"

#: Element classes, keyed by the bridge class name (``type(spec).__name__``).
ELEMENT_PARAMETER_NAMES: dict[str, ElementNames] = {
    "FourNodeQuad": ElementNames(
        owns=frozenset({"pressure"}),
        claims=frozenset({"damp", "material"}),
        forwards=True,
        source=_FOURNODEQUAD,
    ),
    "Truss": ElementNames(
        owns=frozenset({"A", "rho"}),
        claims=frozenset({"material"}),
        forwards=True,
        source="SRC/element/truss/Truss.cpp::Truss::setParameter",
    ),
    "SSPquadUP": ElementNames(
        owns=frozenset({
            "hPerm", "vPerm", "pressureUpperSide", "pressureLowerSide",
            "pressureLeftSide", "pressureRightSide", "b1", "b2"}),
        claims=frozenset(),
        forwards=True,
        source="SRC/element/UWelements/SSPquadUP.cpp::SSPquadUP::setParameter",
    ),
}

_BEAM2D = ("SRC/element/elasticBeamColumn/ElasticBeam2d.cpp::"
           "ElasticBeam2d::setParameter")
_BEAM3D = ("SRC/element/elasticBeamColumn/ElasticBeam3d.cpp::"
           "ElasticBeam3d::setParameter")
_LADRUNO_UP = "SRC/element/ladrunoUP/LadrunoUP.cpp::LadrunoUP::setParameter"

#: Element classes whose names depend on the model's ``ndm``, keyed by the
#: bridge class name, then ``ndm``.  ``elasticBeamColumn`` builds
#: ElasticBeam2d or ElasticBeam3d, which share no section names: ``Iz`` on
#: a 2-D beam, or ``rho`` on a 3-D one, reaches nothing.  ``LadrunoUP``
#: takes ``zPerm`` only at ``ndm=3``.  A model whose ``ndm`` is not here
#: is uncatalogued for the class.
ELEMENT_PARAMETER_NAMES_BY_NDM: dict[str, dict[int, ElementNames]] = {
    "elasticBeamColumn": {
        2: ElementNames(
            owns=frozenset({"E", "A", "I", "rho", "release"}),
            claims=frozenset({"damp"}), forwards=False, source=_BEAM2D),
        3: ElementNames(
            owns=frozenset({"E", "A", "Iz", "Iy", "G", "J",
                            "releasez", "releasey"}),
            claims=frozenset({"damp"}), forwards=False, source=_BEAM3D),
    },
    "LadrunoUP": {
        2: ElementNames(
            owns=frozenset({"xPerm", "yPerm"}),
            claims=frozenset({"material"}), forwards=True,
            source=_LADRUNO_UP),
        3: ElementNames(
            owns=frozenset({"xPerm", "yPerm", "zPerm"}),
            claims=frozenset({"material"}), forwards=True,
            source=_LADRUNO_UP),
    },
}


def element_names(cls: str, ndm: "int | None") -> "ElementNames | None":
    """The catalogue entry of element class ``cls`` in a model of ``ndm``,
    or ``None`` when the class (or, for a class that depends on it, the
    ``ndm``) is not catalogued."""
    entry = ELEMENT_PARAMETER_NAMES.get(cls)
    if entry is not None:
        return entry
    by_ndm = ELEMENT_PARAMETER_NAMES_BY_NDM.get(cls)
    if by_ndm is None or ndm is None:
        return None
    return by_ndm.get(int(ndm))

_SOIL_SOURCES: dict[str, str] = {
    "PressureIndependMultiYield": (
        "SRC/material/nD/soil/PressureIndependMultiYield.cpp::"
        "PressureIndependMultiYield::setParameter"),
    "PressureDependMultiYield": (
        "SRC/material/nD/soil/PressureDependMultiYield.cpp::"
        "PressureDependMultiYield::setParameter"),
    "PressureDependMultiYield02": (
        "SRC/material/nD/soil/PressureDependMultiYield02.cpp::"
        "PressureDependMultiYield02::setParameter"),
    "PM4Sand": "SRC/material/nD/UWmaterials/PM4Sand.cpp::PM4Sand::setParameter",
    "PM4Silt": "SRC/material/nD/UWmaterials/PM4Silt.cpp::PM4Silt::setParameter",
}

#: Material classes, keyed by the bridge class name.  The five soil
#: classes come from ``material/nd_soil.SET_PARAMETER_NAMES``.
MATERIAL_PARAMETER_NAMES: dict[str, MaterialNames] = {
    # ElasticIsotropicPlaneStrain2D / PlaneStress2D / 3D, the copies an
    # element makes, inherit this body.
    "ElasticIsotropic": MaterialNames(
        owns=frozenset({"E", "nu", "v", "rho"}),
        tag_checked=False,
        source=(
            "SRC/material/nD/ElasticIsotropicMaterial.cpp::"
            "ElasticIsotropicMaterial::setParameter"),
    ),
    "ElasticMaterial": MaterialNames(
        owns=frozenset({"E", "Epos", "Eneg", "eta"}),
        tag_checked=False,
        source=(
            "SRC/material/uniaxial/ElasticMaterial.cpp::"
            "ElasticMaterial::setParameter"),
    ),
    "ManzariDafalias": MaterialNames(
        owns=frozenset({
            "updateMaterialStage", "materialState", "IntegrationScheme",
            "Jacobian", "refShearModulus", "ShearModulus", "poissonRatio",
            "voidRatio", "stressCorrection"}),
        tag_checked=True,
        source=(
            "SRC/material/nD/UWmaterials/ManzariDafalias.cpp::"
            "ManzariDafalias::setParameter"),
    ),
    **{
        cls: MaterialNames(owns=names, tag_checked=True,
                           source=_SOIL_SOURCES[cls])
        for cls, names in _SOIL_NAMES.items()
    },
}


@dataclass(frozen=True)
class ParameterTarget:
    """One targeted element spec, as the check reads it: its class, the
    model's ``ndm`` (``None`` before ``ops.model``), and the
    ``(class, tag)`` of each material it carries directly."""

    cls: str
    ndm: "int | None"
    materials: tuple[tuple[str, "int | None"], ...]


#: Uncatalogued classes already warned about, per bridge.
_WARNED: "weakref.WeakKeyDictionary[object, set[str]]" = (
    weakref.WeakKeyDictionary())


def _material_takes(
    entry: MaterialNames, name: str, tag: "int | None",
    mat_tag: "int | None",
) -> bool:
    if name not in entry.owns:
        return False
    if not entry.tag_checked:
        return True
    return mat_tag is not None and tag is not None and int(tag) == mat_tag


def _names(names: Iterable[str]) -> str:
    return ", ".join(sorted(names)) or "none"


def check_parameter_name(
    call: str,
    name: str,
    mat_tag: "int | None",
    targets: Sequence[ParameterTarget],
    *,
    owner: object,
    stacklevel: int = 2,
) -> None:
    """Refuse ``name`` when a targeted catalogued class is known not to take
    it; warn once per uncatalogued class of ``owner`` (the bridge).

    ``call`` opens the messages.  ``mat_tag`` is the ``material=`` tag the
    argv carries, or ``None``.  Each target is checked on its own: one
    refusing class refuses the call.  Raises :class:`BridgeError` before
    any warning, so a refused call warns nothing.
    """
    refused: list[str] = []
    uncatalogued: set[str] = set()
    for target in targets:
        entry = element_names(target.cls, target.ndm)
        if entry is None:
            uncatalogued.add(target.cls)
            continue
        if entry.takes(name):
            continue
        own = (f"{target.cls} takes {_names(entry.owns)}"
               + (f" and any name containing {_names(entry.claims)}"
                  if entry.claims else "")
               + f" ({entry.source})")
        if not entry.forwards:
            refused.append(own + ", and forwards nothing to a material")
            continue
        if not target.materials:
            uncatalogued.add(f"{target.cls} (no material found to forward to)")
            continue
        known = [(m, t, MATERIAL_PARAMETER_NAMES.get(m))
                 for m, t in target.materials]
        if any(e is not None and _material_takes(e, name, t, mat_tag)
               for _m, t, e in known):
            continue
        unknown = {m for m, _t, e in known if e is None}
        if unknown:
            uncatalogued.update(unknown)
            continue
        mats = "; ".join(
            f"{m} takes {_names(e.owns)}"
            + (" (only with its own tag, which material= appends)"
               if e.tag_checked else "")
            + f" ({e.source})"
            for m, _t, e in sorted(known, key=lambda k: k[0])
            if e is not None)
        refused.append(own + f", and forwards the rest to {mats}")
    if refused:
        raise BridgeError(
            f"{call}: no object would take {name!r}, so OpenSees would "
            "print 'no objects were able to identify parameter' and the "
            "update would be lost (issue #2377).  "
            + "  ".join(f"{r}." for r in sorted(set(refused)))
            + "  Use a name the element or its material takes, or target "
            "the elements that own it."
        )
    if not uncatalogued:
        return
    try:
        warned = _WARNED.setdefault(owner, set())
    except TypeError:  # an owner that takes no weak reference
        warned = set()
    fresh = sorted(uncatalogued - warned)
    warned.update(fresh)
    for cls in fresh:
        warnings.warn(
            f"{call}: {cls} is not in the update_parameter name catalogue "
            "(stage/_parameter_names.py), so the bridge cannot tell whether "
            f"its setParameter takes {name!r}.  The update emits; if "
            "OpenSees prints 'no objects were able to identify parameter', "
            "it was lost.  Warned once per class.",
            UncataloguedParameterWarning,
            stacklevel=stacklevel + 1,
        )
