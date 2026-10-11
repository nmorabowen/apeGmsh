"""``_stacklevel_outside_package``: kept at ``_internal/`` depth because its body counts ``dirname`` hops from ``__file__``."""
from __future__ import annotations



def _stacklevel_outside_package() -> int:
    """``stacklevel`` naming the first caller outside ``apeGmsh``.

    Emit is reached at varying depths (``ops.tcl`` -> ``emit``, a direct
    ``ops.build().emit(...)``, the live runners), so a fixed stacklevel
    lands inside the bridge.  Counted from the ``warnings.warn`` call
    in this module's caller.
    """
    import inspect
    import os

    pkg = os.path.normcase(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))) + os.sep
    frame = inspect.currentframe()
    level = 1
    try:
        f = frame.f_back if frame is not None else None  # the warn site
        while f is not None and os.path.normcase(os.path.abspath(
                f.f_code.co_filename)).startswith(pkg):
            f = f.f_back
            level += 1
        return level
    finally:
        del frame
