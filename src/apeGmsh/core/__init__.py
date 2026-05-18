"""apeGmsh.core — geometry / constraint / load / mass composites.

FP-1 import-cycle discipline — see ``tests/test_import_dag_polarity.py``
------------------------------------------------------------------------
``apeGmsh.core`` and ``apeGmsh.mesh`` form a *latent* import cycle:
``core`` modules import ``apeGmsh.mesh`` resolvers/records, and several
``mesh`` modules import ``apeGmsh.core``.  ``import apeGmsh`` survives
this only because every cross-package edge has a deliberate
eager/deferred polarity — the imports that would close the loop eagerly
are kept lazy (function-local or under ``TYPE_CHECKING``), never at
module top level.

Consequences for anyone editing this package:

* This ``__init__`` must stay minimal.  It must NOT eagerly import the
  selection-leaf modules (``_selection``, ``_chain``, ``_spatial``,
  ``_resolution``) — that reopens the cycle and crashes
  ``import apeGmsh``.
* Any NEW module-level ``from apeGmsh.mesh …`` in a ``core`` module
  (or ``from apeGmsh.core …`` in a ``mesh`` module) can flip a deferred
  edge eager with the same effect.  Keep such imports function-local or
  ``TYPE_CHECKING``-gated.

``tests/test_import_dag_polarity.py`` is the tripwire: it freezes the
set of eager cross-package edges and fails CI on any change.  If a
change here is intentional, update its ``BASELINE`` in the same commit
so the import-graph change is an explicit, reviewed diff.
"""
from .Part import Part
from .Model import Model
from ._parts_registry import PartsRegistry, Instance
from .ConstraintsComposite import ConstraintsComposite
from .LoadsComposite import LoadsComposite
from .MassesComposite import MassesComposite

__all__ = [
    "Part", "Model",
    "PartsRegistry", "Instance",
    "ConstraintsComposite",
    "LoadsComposite",
    "MassesComposite",
]
