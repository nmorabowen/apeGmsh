"""Builders the ``.user.h5`` tests share (ADR 0130 VU1).

A plain module, imported as ``tests.results.user._user_helpers``: the folder has
no ``__init__.py``, like its siblings, so it is never a bare top-level ``user``
package (``tests/test_import_namespacing.py``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from apeGmsh.results.user import (
    CaseQuantity,
    CutEntity,
    ElementBucket,
    Entities,
    ModeTable,
    PointSet,
    SpectrumCase,
    StateAddress,
    Storey,
    UserResult,
)

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


def sample_entities() -> Entities:
    """Two cuts (one with a bounding polygon), a storey and a point set."""
    return Entities(
        cuts=(
            CutEntity(name="pier P1 base", point=(0.0, 0.0, 3.0), normal=(0.0, 0.0, 1.0),
                      reference=(0.5, 0.0, 3.0), element_ids=np.array([101, 102], dtype="i4")),
            CutEntity(name="Delta_after (L/480)", point=(2.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                      reference=(2.0, 1.0, 1.5), element_ids=np.array([7], dtype="i4"),
                      side="negative",
                      polygon=np.array([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 3.0]])),
        ),
        storeys=(Storey(name="L1", level=3.0, reference=(1.0, 1.0, 3.0),
                        node_ids=np.array([5, 6, 7, 8], dtype="i4")),),
        point_sets=(PointSet(name="plan points", xyz=np.array([[0.0, 0.0, 3.0], [1.0, 2.0, 3.0]]),
                             host_element=np.array([-1, 102], dtype="i4")),),
    )


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


#: The two eigenvalues of ``apeGmshViewer/fixtures/ladruno_modes.ladruno``.
MODES_LAMBDA = (381.9660112501052, 2618.0339887498953)


def mode_rows(**kw) -> ModeTable:
    """A two-row ``/modes`` table on ``stage_0``, the eigenvalues of the modes fixture."""
    lam = np.array(MODES_LAMBDA)
    fields = dict(
        components=("MX", "MY", "RMZ"), host=("stage_0", "stage_0"), kind=("eigen", "eigen"),
        step=np.array([-1, -1], dtype="i4"), mode=np.array([1, 2], dtype="i4"),
        lam=lam, period=2.0 * np.pi / np.sqrt(lam), generalized_mass=np.array([1.0, np.nan]),
        participation=np.array([[1.2, 0.0, 0.3], [-0.4, 0.0, 0.0]]),
        mass_ratio=np.array([[0.8, 0.0, 0.1], [0.2, 0.0, 0.0]]),
        cumulative_ratio=np.array([[0.8, 0.0, 0.1], [1.0, 0.0, 0.1]]),
    )
    fields.update(kw)
    return ModeTable(**fields)


def case(name: str = "E_x (DBE)", **kw) -> SpectrumCase:
    """A two-mode case: a base shear on the domain and two nodes' displacement."""
    lam = np.array([40.0, 400.0])
    shear = np.array([[[-3.0, 0.5]], [[1.25, -1e-300]]])
    disp = np.arange(8, dtype="f8").reshape(2, 2, 2) / 3.0
    results = (
        CaseQuantity("base_shear", "domain", ("1", "2"), shear),
        CaseQuantity("DISPLACEMENT", "nodes", ("Ux", "Uy"), disp, np.array([4, 7], dtype="i4")),
    )
    reference = (
        CaseQuantity("base_shear", "domain", ("1", "2"), np.array([[[0.0, 9.81]]])),
        CaseQuantity("DISPLACEMENT", "nodes", ("Ux", "Uy"), np.array([[[0.0, -1.0], [0.0, -2.0]]]),
                     np.array([4, 7], dtype="i4")),
    )
    fields = dict(
        name=name, direction=1, scale=0.75,
        spectrum_periods=np.array([0.0, 0.5, 2.0]), spectrum_sa=np.array([3.0, 6.0, 2.0]),
        mode=np.array([1, 2], dtype="i4"), kind=("eigen", "eigen"), lam=lam,
        sa_period=np.zeros(2), gamma=np.array([1.3, -0.4]), sa=np.array([4.5, 3.9]),
        damping=np.array([0.05, np.nan]), results=results, reference=reference,
    )
    fields.update(kw)
    return SpectrumCase(**fields)


def cqc_reference(r: np.ndarray, lam: np.ndarray, xi: np.ndarray, sa_period: np.ndarray) -> np.ndarray:
    """CQC from the formula: modes sharing an Sa period > 0 are one rigid mode."""
    terms: list[np.ndarray] = []
    omegas: list[float] = []
    damping: list[float] = []
    slot: dict[float, int] = {}
    for a in range(lam.size):
        t = float(sa_period[a])
        if t > 0.0 and t in slot:
            terms[slot[t]] = terms[slot[t]] + r[a]
            continue
        if t > 0.0:
            slot[t] = len(terms)
        terms.append(np.array(r[a], dtype=float))
        omegas.append(2 * math.pi / t if t > 0.0 else math.sqrt(lam[a]))
        damping.append(float(xi[a]))
    total = np.zeros_like(terms[0])
    for i, (wi, xi_i, ri) in enumerate(zip(omegas, damping, terms)):
        for j, (wj, xi_j, rj) in enumerate(zip(omegas, damping, terms)):
            if i == j:
                rho = 1.0
            else:
                b = wj / wi
                rho = (8 * math.sqrt(xi_i * xi_j) * (xi_i + b * xi_j) * b ** 1.5
                       / ((1 - b * b) ** 2 + 4 * xi_i * xi_j * b * (1 + b * b)
                          + 4 * (xi_i ** 2 + xi_j ** 2) * b * b))
            total = total + rho * ri * rj
    return np.sqrt(np.clip(total, 0.0, None))
