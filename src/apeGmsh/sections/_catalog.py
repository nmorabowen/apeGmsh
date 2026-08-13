"""apeSteel catalog bridge for the builder's shape picker (ADR 0080 B7).

Qt-free and **fail-soft**: apeSteel is an optional dependency at the GUI
edge, never required headlessly. When it is absent every function here
degrades (``catalog_available()`` is ``False``, ``catalog_labels()`` is
empty, :func:`catalog_shape_params` raises a guided ``ImportError``) and
the builder hides its picker.

Scope is **doubly-symmetric I shapes** — AISC v16 ``W`` / ``S`` / ``HP``
and EN ``IPE`` / ``HE`` — because that is the family apeSteel exposes as
clean plate geometry (``get_doubly_symmetric_i_geometry``), and it maps
1:1 onto the continuum lane's ``W_face(bf, tf, h, tw)`` builder. Other
catalog families (HSS, pipe, channel, angle, tee) have no geometry
accessor to prefill from; author them from their dimensions.

**Units.** apeSteel's native base is N-mm-tonne-s, so prefilled
dimensions are **millimetres**. apeGmsh is unit-agnostic — the numbers
land in the document as authored, and the document's ``units`` label is
the only place that says so.

The ``h`` mapping is the trap this module exists to close:
``W_face``'s ``h`` is the **clear web height** between the flanges, not
the catalog depth ``d``. apeSteel reports exactly that as
``web_clear_height_hw``, so the mapping is direct — but hand-typing
``d`` into the same field is the silent double-flange error.
"""
from __future__ import annotations

from typing import Any


__all__ = [
    "CATALOG_SHAPE",
    "catalog_available",
    "catalog_labels",
    "catalog_shape_params",
]


#: The document shape every catalog prefill targets.
CATALOG_SHAPE = "W_face"

#: apeSteel catalog classes, in picker order (label prefix → class name).
_CATALOGS: tuple[tuple[str, str], ...] = (
    ("AISC v16", "AISCv16Catalog"),
    ("EN", "EuropeanIPECatalog"),
)


def _apesteel() -> Any:
    """Import apeSteel or raise the guided ``ImportError``."""
    try:
        import apeSteel
    except ImportError as exc:
        raise ImportError(
            "the section builder's catalog picker requires the optional "
            "dependency 'apeSteel' (pip install apeSteel). Without it, "
            "type the shape's dimensions into the form directly."
        ) from exc
    return apeSteel


def catalog_available() -> bool:
    """``True`` when apeSteel imports — the picker's visibility gate."""
    try:
        _apesteel()
    except ImportError:
        return False
    return True


def catalog_labels() -> tuple[str, ...]:
    """Every doubly-symmetric I designation the picker can offer.

    **Best-effort.** apeSteel publishes no label-enumeration API, so
    this reads the catalogs' backing tables and returns ``()`` if that
    shape ever changes. An empty result is not an error — the picker
    stays usable as a free-text field, because
    :func:`catalog_shape_params` resolves any designation apeSteel
    accepts whether or not it was listed here.
    """
    try:
        apeSteel = _apesteel()
    except ImportError:
        return ()
    out: list[str] = []
    for _family, cls_name in _CATALOGS:
        cls = getattr(apeSteel, cls_name, None)
        if cls is None:
            continue
        try:
            frame = cls()._dataframe()
            column = frame.columns[
                1 if "AISC_Manual_Label" in frame.columns else 0
            ]
            labels = [str(v) for v in frame[column].tolist()]
        except Exception:       # noqa: BLE001 - best-effort by contract
            continue
        out += [
            label for label in labels
            if label[:1] in ("W", "S", "H", "I") and label.strip()
        ]
    # dedupe, preserving catalog order
    return tuple(dict.fromkeys(out))


def catalog_shape_params(label: str) -> "dict[str, float]":
    """Resolve one designation to ``W_face`` parameters.

    Returns ``{"bf", "tf", "h", "tw"}`` in apeSteel's millimetres, ready
    to splat into :meth:`SectionDocument.add_shape` with
    ``shape=`` :data:`CATALOG_SHAPE`. ``h`` is the **clear web height**
    (``d - 2·tf``), which is what ``W_face`` wants.

    Raises
    ------
    ImportError
        apeSteel is not installed.
    ValueError
        No catalog recognizes ``label``.
    """
    apeSteel = _apesteel()
    errors: list[str] = []
    for family, cls_name in _CATALOGS:
        cls = getattr(apeSteel, cls_name, None)
        if cls is None:
            continue
        try:
            geom = cls().get_doubly_symmetric_i_geometry(label)
        except Exception as exc:    # noqa: BLE001 - try the next catalog
            errors.append(f"{family}: {exc}")
            continue
        return {
            "bf": float(geom.flange_width_bf),
            "tf": float(geom.flange_thickness_tf),
            "h": float(geom.web_clear_height_hw),
            "tw": float(geom.web_thickness_tw),
        }
    raise ValueError(
        f"no apeSteel catalog resolves {label!r} to a doubly-symmetric "
        f"I shape. Tried — " + "; ".join(errors or ["(no catalogs)"])
    )
