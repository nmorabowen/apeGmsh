"""Per-element shell thickness, resolved from the model's section records.

``von_mises_shell`` recovers the extreme-fibre surface stress
σ = N/t ± 6M/t² from the shell stress resultants, so it needs the shell
thickness ``t``. That is a property of the SECTION, not of the results
file, and it is already in the model — this module is the lookup.

**Per element, not per call.** A pane scoped to ``slabs``+``wall`` spans
two thicknesses (0.15 m and 0.25 m on the ssi_frame_wall bench), and a
single scalar would apply one group's thickness to the other — a 2.8×
error in the 6M/t² term, silently. So the answer is a vector aligned to
the slab's ``element_index``.

**Keyed by FEM element id, not the OpenSees tag.** Measured on the
bench: the Gauss slab's ``element_index`` intersects
``ElementRecord.fem_eid`` 864/864 and ``ElementRecord.tag`` 0/864. The
two id spaces are both ints and both plausible, so getting this backwards
produces a silently misaligned thickness vector rather than an error.

Section types understood here, and why only these:

* ``ElasticMembranePlateSection`` — ``params = (E, nu, h, rho)``,
  so ``t = params[2]``.
* ``LayeredShell`` / ``LayeredShellFiberSection`` —
  ``params = (nLayers, matTag1, t1, matTag2, t2, …)`` (see
  ``section/plate.py``'s ``_emit``), so ``t = sum(params[2::2])``, the
  gross through-thickness depth.

Anything else refuses by name rather than guessing. A wrong thickness
does not fail — it scales a stress by (t_wrong/t_right)² and the picture
still looks like a picture.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

__all__ = ["ShellThicknessError", "shell_thickness_by_element"]

#: ``params`` index of ``h`` in ``ElasticMembranePlateSection``
#: (``E, nu, h, rho``).
_PLATE_H_INDEX = 2

_LAYERED_TOKENS = frozenset({"LayeredShell", "LayeredShellFiberSection"})


class ShellThicknessError(ValueError):
    """The shell thickness could not be resolved from the model.

    Distinct from a plain ``ValueError`` so callers that want to DEGRADE
    (the contour gate, which must refuse the slot cleanly) can tell this
    apart from a genuine bad argument.
    """


def _thickness_of_section(record: Any) -> float:
    """Gross through-thickness depth of one shell section record."""
    token = getattr(record, "type_token", None)
    params = getattr(record, "params", None)
    if params is None:
        raise ShellThicknessError(
            f"section tag {getattr(record, 'tag', '?')} ({token}) carries no "
            f"params — a layered/fiber section body is not a shell plate."
        )
    if token == "ElasticMembranePlateSection":
        if len(params) <= _PLATE_H_INDEX:
            raise ShellThicknessError(
                f"ElasticMembranePlateSection tag {record.tag} has "
                f"{len(params)} params, expected at least "
                f"{_PLATE_H_INDEX + 1} (E, nu, h, rho)."
            )
        return float(params[_PLATE_H_INDEX])
    if token in _LAYERED_TOKENS:
        # (nLayers, matTag1, t1, matTag2, t2, …)
        n_layers = int(params[0])
        if len(params) != 1 + 2 * n_layers:
            raise ShellThicknessError(
                f"{token} tag {record.tag} declares {n_layers} layers but "
                f"carries {len(params)} params (expected "
                f"{1 + 2 * n_layers}: nLayers then matTag/thickness pairs)."
            )
        return float(sum(float(t) for t in params[2::2]))
    raise ShellThicknessError(
        f"section tag {getattr(record, 'tag', '?')} is a {token!r}, whose "
        f"through-thickness depth apeGmsh does not know how to read. "
        f"Pass thickness=<t> explicitly."
    )


def shell_thickness_by_element(model: Any) -> "dict[int, float]":
    """``{fem_element_id: thickness}`` for every shell in ``model``.

    Shells whose section cannot be read are OMITTED rather than raised
    over, so one exotic section does not deny the answer for the rest;
    :func:`thickness_vector` decides whether the omission matters for
    the elements actually being drawn.
    """
    sections = {s.tag: s for s in model.sections()}
    out: dict[int, float] = {}
    for element in model.elements():
        if "Shell" not in getattr(element, "type_token", ""):
            continue
        fem_eid = int(getattr(element, "fem_eid", -1))
        if fem_eid < 0:
            continue          # emitted outside a bridge fan-out
        args = getattr(element, "args", ())
        if not args:
            continue
        record = sections.get(int(args[-1]))
        if record is None:
            continue
        try:
            out[fem_eid] = _thickness_of_section(record)
        except ShellThicknessError:
            continue
    return out


def thickness_vector(
    model: Any, element_index: "ndarray",
) -> "ndarray":
    """Per-column thickness aligned to a Gauss slab's ``element_index``.

    Raises :class:`ShellThicknessError` naming the elements it could not
    resolve — never returns a partially-filled vector, because a
    default-filled column is a wrong stress that still draws.
    """
    table = shell_thickness_by_element(model)
    ids = np.asarray(element_index, dtype=np.int64)
    if ids.size == 0:
        return np.empty(0, dtype=np.float64)
    missing = sorted({int(i) for i in ids if int(i) not in table})
    if missing:
        raise ShellThicknessError(
            f"no shell thickness for {len(missing)} element(s) "
            f"(e.g. {missing[:5]}): they are not shells with a readable "
            f"plate/layered section. Pass thickness=<t> explicitly."
        )
    return np.array([table[int(i)] for i in ids], dtype=np.float64)


def is_resolvable(model: Optional[Any]) -> bool:
    """Whether ANY shell thickness can be read from ``model``.

    The cheap question the contour gate asks before offering a
    shell-derived scalar. Deliberately not per element: the gate runs on
    every realize and must not walk the element table each time.
    """
    if model is None:
        return False
    try:
        return bool(shell_thickness_by_element(model))
    except Exception:
        return False
