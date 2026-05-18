"""apeGmsh._kernel — root-leaf pure data/algorithm layer.

This package holds the dependency-free numpy/stdlib payloads, the
pre-mesh definition dataclasses, the resolved record dataclasses, the
defs→records resolvers, the record-set sub-composites, the label-prefix
predicates, and the ``SelectionChain`` mixin.

Layering law (the cycle-break): ``core`` / ``mesh`` / ``viz`` /
``results`` import strictly **downward** into ``_kernel``; ``_kernel``
imports nothing from those packages.  The only permitted upward-looking
dependency is ``apeGmsh.fem`` (leaf-pure shape-function / HRZ helpers),
used by :mod:`apeGmsh._kernel.resolvers._mass_resolver` (plan HT10).

No submodule is re-exported here: every consumer imports the concrete
submodule it needs (``apeGmsh._kernel.payloads``,
``apeGmsh._kernel.defs.loads``, ``apeGmsh._kernel.records``, ...) so the
import graph stays explicit and the tripwire can police it.
"""

from __future__ import annotations
