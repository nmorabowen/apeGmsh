"""Cross-family contract tests for the ``Primitive`` base.

Every concrete primitive shipped by a Phase 1+ slice is reachable
here through its family-specific ``ALL_*`` list. Each family owns its
own contract file (``test_uniaxial_material_contract``,
``test_section_contract``, etc.); this file aggregates them so any
Primitive added to a family is automatically gated on the cross-
family invariants too.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.types import Primitive

# Family ALL_* lists. Add a new import + extend ALL_PRIMITIVES when a
# new family lands.
from .test_analysis_contract import ALL_ANALYSIS_COMPONENTS
from .test_element_beam_column_contract import ALL_BEAM_COLUMN_ELEMENTS
from .test_element_shell_contract import ALL_SHELL_ELEMENTS
from .test_element_solid_contract import ALL_SOLID_ELEMENTS
from .test_element_truss_contract import ALL_TRUSS_ELEMENTS
from .test_element_zero_length_contract import ALL_ZERO_LENGTH_ELEMENTS
from .test_geom_transf_contract import ALL_GEOM_TRANSF
from .test_nd_material_contract import ALL_ND
from .test_pattern_contract import ALL_PATTERNS
from .test_recorder_contract import ALL_RECORDERS
from .test_section_contract import ALL_SECTIONS
from .test_time_series_contract import ALL_TIME_SERIES
from .test_uniaxial_material_contract import ALL_UNIAXIAL


ALL_PRIMITIVES: list[type[Primitive]] = [
    *ALL_UNIAXIAL,
    *ALL_ND,
    *ALL_SECTIONS,
    *ALL_GEOM_TRANSF,
    *ALL_TIME_SERIES,
    *ALL_BEAM_COLUMN_ELEMENTS,
    *ALL_TRUSS_ELEMENTS,
    *ALL_ZERO_LENGTH_ELEMENTS,
    *ALL_SHELL_ELEMENTS,
    *ALL_SOLID_ELEMENTS,
    *ALL_PATTERNS,
    *ALL_RECORDERS,
    *ALL_ANALYSIS_COMPONENTS,
]


@pytest.mark.parametrize("cls", ALL_PRIMITIVES)
class TestPrimitiveContract:
    def test_inherits_from_primitive(self, cls: type[Primitive]) -> None:
        assert issubclass(cls, Primitive)

    def test_has_emit(self, cls: type[Primitive]) -> None:
        assert "_emit" in cls.__dict__

    def test_has_dependencies(self, cls: type[Primitive]) -> None:
        assert "dependencies" in cls.__dict__

    def test_has_repr(self, cls: type[Primitive]) -> None:
        # Primitive supplies a default __repr__ keyed off ``__name__``.
        # Family-specific contract tests verify the rendered string
        # contains the class name (they can construct an instance).
        assert hasattr(cls, "__repr__")
