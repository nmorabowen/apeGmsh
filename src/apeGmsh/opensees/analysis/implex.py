"""
The IMPL-EX time driver — ADR 0113 slice 1.

ASDConcrete3D / ASDConcrete1D extrapolate their internal variables with
the ratio ``dtime_n / dtime_n_commit`` (IMPL-EX) and scale their
viscosity with ``dtime_n`` (``eta``).  Each material reads ``dtime_n``
from OpenSees' own increment (``ops_Dt``) **until the first**
``setParameter`` / ``updateParameter`` of ``dTime``, ``dTimeCommit`` or
``dTimeInitial`` reaches it; from then on ``dtime_is_user_defined`` stays
true and the material keeps whatever value was last written
(``ASDConcrete3DMaterial.cpp``: ``if (!dtime_is_user_defined) dtime_n =
ops_Dt``; ``updateParameter`` cases 2000-2002).  STKO writes the three
values before every attempted increment
(``STKO_DT_UTIL_OnBeforeAnalyze``); a deck that writes them once and then
changes the increment runs on a stale ``dTime``.

:class:`ImplexTime` is the typed declaration of that driver, on the
bridge as ``ops.implex_time(mode=...)``:

* ``mode="stko"`` -- STKO's hook: three persistent parameters over every
  element whose material closure reaches an IMPL-EX (or ``eta > 0``)
  ASDConcrete, and the increment written before every stage's analyze
  loop (``dTimeCommit`` / ``dTimeInitial`` at the stage's first
  increment too).
* ``mode="off"`` -- no driver, and nothing may write ``dTime*``: the
  materials follow ``ops_Dt`` for the whole run.  Declaring it makes the
  choice explicit and lets the bridge refuse a deck that contradicts it.
* ``mode="follow"`` -- reserved.  Following ``ops_Dt`` while resetting
  the IMPL-EX ratio at a stage start needs an ASDConcrete parameter that
  does not exist yet (ADR 0113 D8, fork dependency F1); refused.

The declaration is not a registered primitive (nothing to tag or order);
it parameterizes the staged emit, like a ``Ladder`` parameterizes an
analyze loop.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from .._internal.types import Primitive

__all__ = [
    "DTIME_PARAMETERS",
    "ImplexMode",
    "ImplexTime",
    "implex_element_specs",
    "material_closure",
    "reads_dtime",
]

#: The three ASDConcrete time parameters the driver writes, in the order
#: of its persistent parameters (``dTime`` first: it is written on every
#: call; the other two only at a stage's first increment).
DTIME_PARAMETERS: tuple[str, str, str] = ("dTime", "dTimeCommit", "dTimeInitial")

ImplexMode = Literal["stko", "off", "follow"]

_MODES: frozenset[str] = frozenset({"stko", "off", "follow"})


@dataclass(frozen=True)
class ImplexTime:
    """Model-wide IMPL-EX ``dTime`` driver declaration (ADR 0113 D1).

    Parameters
    ----------
    mode
        ``"stko"`` (drive ``dTime`` as STKO does), ``"off"`` (no driver;
        any ``dTime*`` write is refused) or ``"follow"`` (reserved;
        refused until the fork parameter of ADR 0113 D8 F1 exists).
    """

    mode: ImplexMode = "stko"

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(
                f"ImplexTime: mode must be one of {sorted(_MODES)}, got "
                f"{self.mode!r}."
            )
        if self.mode == "follow":
            raise ValueError(
                "ImplexTime(mode='follow') is reserved (ADR 0113 D8, fork "
                "dependency F1): following OpenSees' increment while "
                "resetting the IMPL-EX ratio at a stage start needs an "
                "ASDConcrete parameter that resets dTimeCommit without "
                "setting dtime_is_user_defined, which no OpenSees build "
                "has.  Use mode='stko' (STKO's driver) or mode='off'."
            )

    @property
    def drives(self) -> bool:
        """True iff this declaration emits a driver (``mode="stko"``)."""
        return self.mode == "stko"


def reads_dtime(prim: object) -> bool:
    """True iff ``prim`` is a material whose response reads ``dTime``.

    ASDConcrete3D / ASDConcrete1D with ``implex=True`` (the IMPL-EX
    extrapolation ratio) or ``eta > 0`` (the viscous regularization
    ``eta / (eta + dtime_n)``), and ASDSteel1D with ``implex=True`` (same
    ``dtime_is_user_defined`` switch, ``ASDSteel1DMaterial.cpp``
    ``setTrialStrain`` / ``updateParameter``; it has no ``eta``) --
    STKO's C13 predicate restricted to the types the bridge has.
    ``DamageTC1D/3D`` and ``ASDBondSlip`` join the predicate when the
    bridge types them.
    """
    from ..material.nd import ASDConcrete3D
    from ..material.uniaxial import ASDConcrete1D, ASDSteel1D

    if isinstance(prim, (ASDConcrete3D, ASDConcrete1D)):
        return bool(prim.implex) or float(prim.eta) > 0.0
    if isinstance(prim, ASDSteel1D):
        return bool(prim.implex)
    return False


def material_closure(prim: Primitive) -> Iterator[Primitive]:
    """Every primitive reachable from ``prim`` through ``dependencies()``
    (``prim`` included), each once, depth first.

    An element reaches its section, a beam integration its sections, a
    fiber section its fiber materials, a ``PlateFromPlaneStress`` its
    plane-stress law: the same edges the topological emit order uses
    (charter P5), so a target is exactly an element whose emitted
    definition chain names an IMPL-EX material.
    """
    seen: set[int] = set()
    stack: list[Primitive] = [prim]
    while stack:
        p = stack.pop()
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield p
        stack.extend(reversed(tuple(p.dependencies())))


def implex_element_specs(elements: Iterable[Primitive]) -> list[Primitive]:
    """The element specs (in the given order) whose closure reaches a
    material that :func:`reads_dtime` (ADR 0113 D2)."""
    return [
        e for e in elements
        if any(reads_dtime(p) for p in material_closure(e))
    ]
