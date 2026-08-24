"""Shell thickness resolved per element, so ``von_mises_shell`` needs no argument.

``von_mises_shell`` recovers σ = N/t ± 6M/t² and used to demand an
explicit ``thickness=``. A slot carries a bare component name and
nothing else, so shells could be contoured on their raw resultants but
never on the one scalar an engineer wants.

The thickness is a property of the SECTION and is already in the model.
Two things make the lookup worth testing rather than eyeballing:

* **It must be per ELEMENT.** A read can span groups of different
  thickness — the ssi_frame_wall bench has 0.15 m slabs and a 0.25 m
  wall — and one scalar there applies one group's thickness to the
  other. That is a (t_wrong/t_right)² error in the 6M/t² term which
  still draws a plausible picture, so a single-thickness fixture would
  pass with the bug intact. Every test here uses TWO thicknesses.
* **It is keyed by FEM element id, not the OpenSees tag.** Measured on
  the bench: a Gauss slab's ``element_index`` intersects
  ``ElementRecord.fem_eid`` 864/864 and ``.tag`` 0/864. Both are ints,
  both plausible; getting it backwards silently misaligns the vector.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import _derived, _shell_thickness
from apeGmsh.results._shell_thickness import ShellThicknessError


# =====================================================================
# Model stubs — only what the resolver reads
# =====================================================================

def _section(token: str, tag: int, params: tuple) -> SimpleNamespace:
    return SimpleNamespace(type_token=token, tag=tag, params=params)


def _element(token: str, tag: int, sec_tag: int, fem_eid: int) -> SimpleNamespace:
    return SimpleNamespace(
        type_token=token, tag=tag, args=(sec_tag,), fem_eid=fem_eid,
    )


def _model(sections, elements) -> SimpleNamespace:
    return SimpleNamespace(
        sections=lambda: tuple(sections), elements=lambda: tuple(elements),
    )


#: The bench's own numbers: a thin slab and a thicker wall.
T_SLAB, T_WALL = 0.15, 0.25


def _two_thickness_model() -> SimpleNamespace:
    return _model(
        sections=[
            _section("ElasticMembranePlateSection", 4, (25e6, 0.2, T_SLAB, 2.4)),
            _section("ElasticMembranePlateSection", 5, (25e6, 0.2, T_WALL, 2.4)),
        ],
        elements=[
            _element("ASDShellQ4", 8209, 4, 1001),
            _element("ASDShellQ4", 8210, 4, 1002),
            _element("ASDShellQ4", 8211, 5, 2001),
        ],
    )


# =====================================================================
# The section-type policy
# =====================================================================

def test_plate_section_thickness_is_the_third_param() -> None:
    """``ElasticMembranePlateSection`` params are ``(E, nu, h, rho)``."""
    table = _shell_thickness.shell_thickness_by_element(_two_thickness_model())
    assert table == {1001: T_SLAB, 1002: T_SLAB, 2001: T_WALL}


@pytest.mark.parametrize(
    "token", ["LayeredShell", "LayeredShellFiberSection"],
)
def test_layered_thickness_sums_the_layers(token: str) -> None:
    """``params = (nLayers, matTag1, t1, matTag2, t2, …)``.

    Encoding read off ``section/plate.py``'s ``_emit``, not guessed.
    """
    model = _model(
        sections=[_section(token, 9, (3, 1, 0.05, 2, 0.10, 1, 0.05))],
        elements=[_element("ASDShellQ4", 1, 9, 500)],
    )
    table = _shell_thickness.shell_thickness_by_element(model)
    assert table[500] == pytest.approx(0.20)


def test_layered_arity_mismatch_is_refused() -> None:
    """A declared layer count that does not match the params.

    Summing ``params[2::2]`` regardless would return a plausible number
    off a malformed record.
    """
    model = _model(
        sections=[_section("LayeredShell", 9, (3, 1, 0.05))],   # says 3, has 1
        elements=[_element("ASDShellQ4", 1, 9, 500)],
    )
    assert _shell_thickness.shell_thickness_by_element(model) == {}
    with pytest.raises(ShellThicknessError, match="declares 3 layers"):
        _shell_thickness._thickness_of_section(
            _section("LayeredShell", 9, (3, 1, 0.05)),
        )


def test_unknown_section_type_refuses_by_name() -> None:
    """Never guess a thickness — a wrong one still draws."""
    with pytest.raises(ShellThicknessError, match="Fiber"):
        _shell_thickness._thickness_of_section(
            _section("Fiber", 7, (1.0, 2.0, 3.0)),
        )


def test_elements_without_a_fem_id_are_skipped() -> None:
    """``fem_eid`` is ``-1`` for records emitted outside a bridge fan-out.

    The slab's ``element_index`` is in FEM id space, so a ``-1`` cannot
    be addressed and must not land in the table under that key.
    """
    model = _model(
        sections=[_section("ElasticMembranePlateSection", 4, (25e6, 0.2, T_SLAB, 2.4))],
        elements=[_element("ASDShellQ4", 8209, 4, -1)],
    )
    assert _shell_thickness.shell_thickness_by_element(model) == {}


def test_non_shell_elements_are_ignored() -> None:
    model = _model(
        sections=[_section("Elastic", 1, (25e6, 0.25, 1.0, 1.0, 1.0, 1.0))],
        elements=[_element("forceBeamColumn", 3, 1, 700)],
    )
    assert _shell_thickness.shell_thickness_by_element(model) == {}


# =====================================================================
# The vector, aligned to a slab
# =====================================================================

def test_thickness_vector_follows_element_index() -> None:
    """One entry per COLUMN, in the slab's own order — repeats included.

    A Gauss slab has one column per (element, gauss point), so the same
    element id recurs and the vector has to repeat with it.
    """
    model = _two_thickness_model()
    element_index = np.array([1001, 1001, 2001, 2001, 1002], dtype=np.int64)
    vec = _shell_thickness.thickness_vector(model, element_index)
    assert vec.tolist() == [T_SLAB, T_SLAB, T_WALL, T_WALL, T_SLAB]


def test_thickness_vector_refuses_rather_than_part_filling() -> None:
    """An unresolved element must not get a default.

    A default-filled column is a wrong stress that still renders, which
    is worse than a refusal.
    """
    model = _two_thickness_model()
    with pytest.raises(ShellThicknessError, match="no shell thickness for"):
        _shell_thickness.thickness_vector(
            model, np.array([1001, 999999], dtype=np.int64),
        )


def test_is_resolvable_is_false_without_a_model() -> None:
    assert _shell_thickness.is_resolvable(_two_thickness_model())
    assert not _shell_thickness.is_resolvable(None)
    assert not _shell_thickness.is_resolvable(_model([], []))


# =====================================================================
# compute_shell with a per-column thickness
# =====================================================================

def _resultants(n_cols: int, *, N: float, M: float) -> dict[str, np.ndarray]:
    """Uniaxial N and M on every column, one time step."""
    zero = np.zeros((1, n_cols))
    return {
        "membrane_force_xx": np.full((1, n_cols), N),
        "membrane_force_yy": zero.copy(),
        "membrane_force_xy": zero.copy(),
        "bending_moment_xx": np.full((1, n_cols), M),
        "bending_moment_yy": zero.copy(),
        "bending_moment_xy": zero.copy(),
    }


def test_per_column_thickness_is_not_one_scalar_applied_twice() -> None:
    """The whole point of the slice.

    Two columns, two thicknesses, one call. Each column must equal the
    scalar answer for ITS OWN thickness — and the two must differ, or
    the test would pass on a mixed-thickness bug.
    """
    N, M = 50.0, 10.0
    cols = _resultants(2, N=N, M=M)
    mixed = _derived.compute_shell(
        "von_mises_shell", cols, thickness=np.array([T_SLAB, T_WALL]),
    )
    thin = _derived.compute_shell(
        "von_mises_shell", _resultants(1, N=N, M=M), thickness=T_SLAB,
    )
    thick = _derived.compute_shell(
        "von_mises_shell", _resultants(1, N=N, M=M), thickness=T_WALL,
    )
    assert thin[0, 0] != pytest.approx(thick[0, 0]), (
        "precondition: the two thicknesses must give different answers, "
        "else this test cannot detect a mixed-thickness bug"
    )
    assert mixed[0, 0] == pytest.approx(thin[0, 0])
    assert mixed[0, 1] == pytest.approx(thick[0, 0])


def test_per_column_thickness_matches_the_closed_form() -> None:
    """σ = N/t ± 6M/t², envelope = max(|top|, |bottom|), per column."""
    N, M = 50.0, 10.0
    mixed = _derived.compute_shell(
        "von_mises_shell", _resultants(2, N=N, M=M),
        thickness=np.array([T_SLAB, T_WALL]),
    )
    for col, t in enumerate((T_SLAB, T_WALL)):
        expected = max(
            abs(N / t + 6.0 * M / t**2), abs(N / t - 6.0 * M / t**2),
        )
        assert mixed[0, col] == pytest.approx(expected)


def test_scalar_thickness_still_works() -> None:
    """The explicit override is the escape hatch for a section apeGmsh
    cannot read, so it must not regress."""
    out = _derived.compute_shell(
        "von_mises_shell", _resultants(3, N=50.0, M=0.0), thickness=0.2,
    )
    assert out.shape == (1, 3)
    assert np.allclose(out, 50.0 / 0.2)


def test_misaligned_thickness_vector_is_refused() -> None:
    """A length mismatch would broadcast-or-crash somewhere deeper."""
    with pytest.raises(ValueError, match="align to the slab"):
        _derived.compute_shell(
            "von_mises_shell", _resultants(3, N=1.0, M=0.0),
            thickness=np.array([0.1, 0.2]),
        )


def test_non_positive_thickness_is_refused_per_column() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        _derived.compute_shell(
            "von_mises_shell", _resultants(2, N=1.0, M=0.0),
            thickness=np.array([0.1, 0.0]),
        )
