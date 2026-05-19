from .Inspect import Inspect

# selection-unification v2 P3-R / SC-8 + R-v2-8: the ``viz.Selection``
# *class* is RETAINED (viewer pick-result type, built via deferred
# in-method import in the viewers); only its package export is dropped.
__all__ = ["Inspect"]
