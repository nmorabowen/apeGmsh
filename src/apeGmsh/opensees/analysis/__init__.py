"""
``apeGmsh.opensees.analysis`` — typed primitives for the OpenSees
analysis-chain command families.

Phase 3C ships seven family modules, mirroring the OpenSees source
layout under ``SRC/analysis/``:

* :mod:`.constraint_handler` — ``constraints <Type>`` (Plain, Penalty,
  Transformation, Lagrange)
* :mod:`.numberer`            — ``numberer <Type>`` (Plain, RCM, AMD)
* :mod:`.system`              — ``system <Type>`` (BandGeneral,
  BandSPD, ProfileSPD, UmfPack, Mumps, SparseGeneral, FullGeneral)
* :mod:`.test`                — ``test <Type>`` (NormDispIncr,
  NormUnbalance, EnergyIncr, FixedNumIter, RelativeNormDispIncr)
* :mod:`.algorithm`           — ``algorithm <Type>`` (Linear, Newton,
  ModifiedNewton, NewtonLineSearch, KrylovNewton, BFGS, Broyden)
* :mod:`.integrator`          — ``integrator <Type>`` (LoadControl,
  DisplacementControl, ArcLength, Newmark, HHT, CentralDifference,
  ExplicitDifference)
* :mod:`.analysis`            — ``analysis <Type>`` (Static, Transient,
  VariableTransient)

The ``analyze N [dt]`` driver lives on the bridge as
``apeSees.analyze(steps=, dt=)`` — see ``architecture/api-design.md``.
The ``analysis`` directive only configures which OpenSees Analysis
subclass is instantiated.

Constructing these classes outside a bridge is supported (P11). The
typed methods that auto-register them live on
:mod:`apeGmsh.opensees._internal.ns.analysis` and are exposed via the
seven bridge namespaces (``ops.constraints.X(...)`` etc.).

Two classes named ``Plain`` ship in this package — one for
``constraints``, one for ``numberer``. Re-export them under
unambiguous names (:class:`PlainConstraints`,
:class:`PlainNumberer`) so cross-family code can refer to either
without a per-module import.
"""
from __future__ import annotations

from .algorithm import (
    BFGS,
    Broyden,
    KrylovNewton,
    Linear,
    LineSearchType,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
    NewtonTangent,
)
from .analysis import Static, Transient, VariableTransient
from .constraint_handler import Lagrange, Penalty, Transformation
from .eigen import EigenResult
from .constraint_handler import Plain as PlainConstraints
from .integrator import (
    ArcLength,
    CentralDifference,
    DisplacementControl,
    ExplicitDifference,
    HHT,
    LoadControl,
    Newmark,
)
from .numberer import AMD, RCM
from .numberer import Plain as PlainNumberer
from .system import (
    BandGeneral,
    BandSPD,
    FullGeneral,
    Mumps,
    ProfileSPD,
    SparseGeneral,
    UmfPack,
)
from .test import (
    EnergyIncr,
    FixedNumIter,
    NormDispIncr,
    NormUnbalance,
    RelativeNormDispIncr,
)


__all__ = [
    # constraint_handler
    "PlainConstraints",
    "Penalty",
    "Transformation",
    "Lagrange",
    # numberer
    "PlainNumberer",
    "RCM",
    "AMD",
    # system
    "BandGeneral",
    "BandSPD",
    "ProfileSPD",
    "UmfPack",
    "Mumps",
    "SparseGeneral",
    "FullGeneral",
    # test
    "NormDispIncr",
    "NormUnbalance",
    "EnergyIncr",
    "FixedNumIter",
    "RelativeNormDispIncr",
    # algorithm
    "Linear",
    "Newton",
    "ModifiedNewton",
    "NewtonLineSearch",
    "KrylovNewton",
    "BFGS",
    "Broyden",
    "NewtonTangent",
    "LineSearchType",
    # integrator
    "LoadControl",
    "DisplacementControl",
    "ArcLength",
    "Newmark",
    "HHT",
    "CentralDifference",
    "ExplicitDifference",
    # analysis
    "Static",
    "Transient",
    "VariableTransient",
    # eigen (one-shot, returns values)
    "EigenResult",
]
