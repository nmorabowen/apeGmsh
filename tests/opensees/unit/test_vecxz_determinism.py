"""The orientation-derived ``vecxz`` is bit-reproducible.

The vecxz is written into the deck, so its last bit must not depend on the
host, the BLAS build or where numpy happened to allocate the inputs. The
golden arch emitted ``0.25881904510252085 ... 0.9659258262890684`` on one
host and ``0.2588190451025208 ... 0.9659258262890682`` on another, and in a
different element-tag mode on the same CI runner (#1279): ``np.dot`` /
``np.linalg.norm`` round by CPU dispatch and alignment. The orientation
math now runs on Python floats in a fixed order (``_orientation._dot3`` and
friends); these tests pin that.

The golden corpus cannot catch a regression here on its own: it compares
decks within a last-ulp float tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh.opensees import _orientation
from apeGmsh.opensees._internal.build import compute_vecxz_for_element
from apeGmsh.opensees._orientation import (
    AlongBeam,
    Cartesian,
    Cylindrical,
    Spherical,
    resolve_vecxz,
)
from apeGmsh.opensees.transform import Linear


def _bits(v: tuple[float, float, float]) -> tuple[str, str, str]:
    return tuple(float(x).hex() for x in v)  # type: ignore[return-value]


# The arch of the golden fixture ``arch_with_orientation_fan_out``, with its
# coordinates as literals so this test involves no libm call.
_ARCH = [
    (1.0, 0.0, 0.0),
    (0.8660254037844387, 0.0, 0.49999999999999994),
    (0.5000000000000001, 0.0, 0.8660254037844386),
    (0.0, 0.0, 1.0),
]
_ARCH_VECXZ = [
    (0.9659258262890683, 0.0, 0.2588190451025207),
    (0.7071067811865476, 0.0, 0.7071067811865475),
    (0.2588190451025208, 0.0, 0.9659258262890682),
]


def test_arch_vecxz_bits_are_pinned() -> None:
    transf = Linear(orientation=Spherical())
    got = [
        compute_vecxz_for_element(transf, _ARCH[k], _ARCH[k + 1])
        for k in range(3)
    ]
    assert [_bits(v) for v in got] == [_bits(v) for v in _ARCH_VECXZ]


# ---------------------------------------------------------------------------
# A set of rotated elements, evaluated through many allocation patterns
# ---------------------------------------------------------------------------

def _rotated_elements() -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Unit-ish elements rotated about several axes, off every origin."""
    out = []
    for k in range(24):
        a = 0.37 * k + 0.1
        b = 0.23 * k - 0.4
        d = (
            math.cos(a) * math.cos(b),
            math.sin(a) * math.cos(b),
            math.sin(b),
        )
        p = (1.5 + 0.1 * k, -0.7 + 0.05 * k, 2.0 - 0.03 * k)
        out.append((p, (p[0] + 1.3 * d[0], p[1] + 1.3 * d[1], p[2] + 1.3 * d[2])))
    return out


_ORIENTATIONS = {
    "cartesian": lambda: Cartesian(reference_axis=(0.3, -0.2, 1.0)),
    "cylindrical": lambda: Cylindrical(origin=(0.1, 0.2, 0.0), axis=(0.0, 0.1, 1.0)),
    "spherical": lambda: Spherical(origin=(0.2, -0.1, 0.3)),
}


def _misaligned(v: tuple[float, ...]) -> np.ndarray:
    """A float64 view whose data starts 8 bytes past a 16-byte boundary."""
    buf = np.zeros(8, dtype=np.float64)
    base = buf.ctypes.data % 16
    start = 1 if base == 0 else 0
    out = buf[start:start + 3]
    out[:] = v
    return out


def _strided(v: tuple[float, ...]) -> np.ndarray:
    """A non-contiguous column view of a Fortran-ordered block."""
    block = np.zeros((7, 3), dtype=np.float64, order="F")
    block[4, :] = v
    return block[4, :]


def _after_churn(v: tuple[float, ...], n: int) -> np.ndarray:
    """An array allocated after ``n`` doubles of heap churn."""
    _junk = [np.ones(m) for m in (n, 3 * n + 1, 7)]
    out = np.array(v, dtype=float)
    del _junk
    return out


_PATHS = {
    "tuple": lambda v: tuple(v),
    "list": lambda v: list(v),
    "array": lambda v: np.array(v),
    "misaligned": _misaligned,
    "strided": _strided,
    "churn-1": lambda v: _after_churn(v, 1),
    "churn-1000": lambda v: _after_churn(v, 1000),
    "churn-100003": lambda v: _after_churn(v, 100_003),
    "slice-of-big": lambda v: np.concatenate([np.zeros(4097), v])[4097:],
}


@pytest.mark.parametrize("kind", sorted(_ORIENTATIONS))
@pytest.mark.parametrize("roll", [0.0, 30.0])
def test_vecxz_identical_across_allocation_patterns(kind: str, roll: float) -> None:
    transf = Linear(orientation=_ORIENTATIONS[kind](), roll_deg=roll)
    elems = _rotated_elements()
    ref = [_bits(compute_vecxz_for_element(transf, a, b)) for a, b in elems]
    for name, wrap in _PATHS.items():
        got = [
            _bits(compute_vecxz_for_element(transf, wrap(a), wrap(b)))
            for a, b in elems
        ]
        assert got == ref, f"{kind}, roll={roll}: {name} differs"


def test_resolve_vecxz_identical_for_array_and_tuple_triads() -> None:
    orient = Spherical(origin=(0.2, -0.1, 0.3))
    for a, b in _rotated_elements():
        e = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        n = math.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
        t = (e[0] / n, e[1] / n, e[2] / n)
        triad = orient.triad_at(a)
        as_arrays = resolve_vecxz(np.array(t), *triad)
        as_tuples = resolve_vecxz(t, *(tuple(map(float, x)) for x in triad))
        assert _bits(as_arrays) == _bits(as_tuples)


# ---------------------------------------------------------------------------
# Structural guard: no numpy reduction on the vecxz path
# ---------------------------------------------------------------------------

class _FakeFem:
    """Just enough FEMData for ``AlongBeam.bind_fem``."""

    class _Nodes:
        coords = np.array([
            (0.0, 0.0, 0.0), (1.0, 0.2, 0.1), (2.0, 0.5, 0.1), (3.0, 0.9, 0.0),
        ])

        def index(self, nid: int) -> int:
            return nid - 1

    class _Elements:
        def select(self, pg: str) -> "_FakeFem._Elements":
            return self

        def groups(self) -> list[list[tuple[int, tuple[int, int]]]]:
            return [[(1, (1, 2)), (2, (2, 3)), (3, (3, 4))]]

    nodes = _Nodes()
    elements = _Elements()


def test_vecxz_path_uses_no_numpy_reductions(monkeypatch: pytest.MonkeyPatch) -> None:
    along = AlongBeam(reference_pg="Ref")
    along.bind_fem(_FakeFem())
    orientations = [f() for f in _ORIENTATIONS.values()] + [along]

    def _banned(*_a: object, **_k: object) -> None:
        raise AssertionError("numpy reduction on the vecxz path")

    for name in ("dot", "cross", "inner", "vdot"):
        monkeypatch.setattr(_orientation.np, name, _banned)
    monkeypatch.setattr(_orientation.np.linalg, "norm", _banned)

    for orient in orientations:
        for roll in (0.0, 15.0):
            transf = Linear(orientation=orient, roll_deg=roll)
            for a, b in _rotated_elements()[:6]:
                compute_vecxz_for_element(transf, a, b)
