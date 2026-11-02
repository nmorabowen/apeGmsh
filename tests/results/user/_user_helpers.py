"""Builders the ``.user.h5`` tests share (ADR 0130 VU1).

A plain module, imported as ``tests.results.user._user_helpers``: the folder has
no ``__init__.py``, like its siblings, so it is never a bare top-level ``user``
package (``tests/test_import_namespacing.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apeGmsh.results.user import ElementBucket, StateAddress, UserResult

REPO = Path(__file__).resolve().parents[3]
LADRUNO = REPO / "tests" / "fixtures" / "ladruno"
FEM_HASH = "fedcba9876543210fedcba9876543210"
MODEL_HASH = "0123456789abcdef0123456789abcdef"


def nodal(name: str = "Delta_after (L/480)", *, basis: str = "model", **kw) -> UserResult:
    """A two-state nodal field with a NaN cell: ``D (2, 3, 2)``."""
    data = np.array(
        [[[1.5, -2.5], [np.nan, 1e-300], [3.0, 4.0]],
         [[0.1, 0.2], [0.3, 0.4], [np.nan, np.nan]]], dtype="f8",
    )
    fields = dict(
        name=name, target="nodes", basis=basis,
        states=(StateAddress("stage", "gravity", -1, 0), StateAddress("combination", "1.2D+1.6L")),
        components=("wx", "wy"), units=("mm", "mm"), data=data,
        ids=np.array([1, 2, 3], dtype="i4"),
    )
    fields.update(kw)
    return UserResult(**fields)


def bucket(**kw) -> ElementBucket:
    """A barycentric shell bucket: two elements, two Gauss points, three moments."""
    data = np.arange(12, dtype="f8").reshape(1, 2, 6) / 7.0
    fields = dict(
        name="204-ASDShellT3[102:0:0]", ids=np.array([11, 12], dtype="i4"), data=data,
        components=(("mxx", "myy", "mxy"),), gauss_ids=(0,), multiplicity=(2,),
        topology="tri", param_domain="bary", family="lagrange", order=(1, 1),
        gp_param=np.array([[0.2, 0.3], [0.1, 0.6]]),
    )
    fields.update(kw)
    return ElementBucket(**fields)


def elements(b: ElementBucket) -> UserResult:
    return UserResult(name="wood_armer", target="elements", basis="model", buckets=(b,))
