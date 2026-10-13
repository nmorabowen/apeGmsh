"""Stage-bound analysis builder (private; see ``_builder``).

``StageChain`` is the shared analysis chain ``ops.analysis.chain(...)``
returns; it is public so a script can annotate it.
"""

from ._chain import StageChain

__all__ = ["StageChain"]
