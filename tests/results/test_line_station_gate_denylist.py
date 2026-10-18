"""The line-station "never a station source" gate is a deny-list (#2119).

Before #2119 a ``globalForce``-only bucket warned only for the beam classes
the response catalog knows (``NODAL_FORCE_CATALOG`` and
``CUSTOM_RULE_CATALOG``). A beam it does not know (``ModElasticBeam3d``,
``MixedBeamColumn3d``, ``ForceBeamColumnCBDI3d``, ``Pipe``, ...) read ``[]``
with no word on both formats. Now the gate names every class but a
non-beam: the continuum, shell and truss classes of ``RESPONSE_CATALOG``,
the shells of ``LAYER_CATALOG``, the springs of ``ZEROLENGTH_CATALOG``, and
the bearings, walls and links of ``NON_BEAM_LINE_CLASSES``.

The oracle per bucket is the warning itself: a beam class's ``globalForce``
only bucket names the bucket and its elements and reads no stations; a
non-beam's reads no stations and warns nothing. The bucket is #2107's
cantilever element 11 (``tests/test_results_line_station_source_rank.py``),
whose ``globalForce`` names ``Mz_1``/``My_1`` as every two-node element does.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees._response_catalog import (
    CUSTOM_RULE_CATALOG,
    FIBER_CATALOG,
    LAYER_CATALOG,
    NODAL_FORCE_CATALOG,
    NON_BEAM_LINE_CLASSES,
    RESPONSE_CATALOG,
    ZEROLENGTH_CATALOG,
)
from apeGmsh.results.readers._mpco_local_force_io import (
    LINE_STATION_NON_BEAM_CLASSES,
)
from tests.test_results_line_station_source_rank import (
    FORMATS,
    SKIP_WARNING,
    _File,
    _read,
)

MOMENTS = ("bending_moment_z", "bending_moment_y")

#: One beam per family outside the catalog's beam tables (the issue's
#: list, by the fork's ``getClassType()``). Each must warn by name.
UNCATALOGUED_BEAMS = [
    "4-ModElasticBeam3d[1:0:0]",                 # elastic, modified
    "30-MixedBeamColumn3d[1000:1:0]",            # mixed formulation
    "31-MixedBeamColumnAsym3d[1000:1:0]",        # mixed, asymmetric
    "78-ForceBeamColumnCBDI3d[1000:1:0]",        # force, CBDI
    "84-ForceBeamColumn3dThermal[1000:1:0]",     # force, thermal
    "65-DispBeamColumnAsym3d[1000:1:0]",         # displacement, asymmetric
    "149-ElasticBeamWarping3d[1:0:0]",           # elastic, warping
    "260-Pipe[1:0:0]",                           # pipe
    "33015-LadrunoIMKBeam[1:0:0]",               # fork IMK beam
]

#: One class per non-beam family, each from the table that derives it.
#: A read of each stays silent.
NON_BEAMS = [
    "149-ElastomericBearingPlasticity3d[1:0:0]",  # bearing (elastomeric)
    "190-SingleFPSimple3d[1:0:0]",                 # bearing (friction)
    "212-MVLEM3d[1:0:0]",                          # wall
    "213-SFI_MVLEM3d[1:0:0]",                      # wall (SFI)
    "86-TwoNodeLink[1:0:0]",                       # link
    "19-ZeroLength[1:0:0]",                        # ZEROLENGTH_CATALOG
    "53-ShellMITC4[201:0:0]",                      # LAYER_CATALOG shell
    "56-Brick[401:0:0]",                           # RESPONSE_CATALOG solid
    "12-Truss[1:0:0]",                             # RESPONSE_CATALOG truss
]

#: Two-node classes the deny-list leaves out on purpose: joints and
#: cables. Fail closed: they are named, as an unknown class is.
NAMED_NON_BEAMS = [
    "71-BeamColumnJoint3d[1:0:0]",
    "82-Joint3D[1:0:0]",
    "300-CatenaryCable[1:0:0]",
]


def _class(key: str) -> str:
    return key.split("-", 1)[1].split("[", 1)[0]


def _assert_names(fmt: str, path: Path, key: str, comp: str) -> None:
    f = _File(fmt, path)
    f.add("globalForce", [11], key=key)
    with f.reader() as r:
        for ids in ([11], None):
            with pytest.warns(
                SKIP_WARNING[fmt],
                match=rf"globalForce/{re.escape(key)}.*\[11\].*never a "
                      rf"line-station source",
            ):
                slab = _read(r, comp, ids)
            assert slab.element_index.size == 0


# ---------------------------------------------------------------------
# A beam the catalog does not know warns by name
# ---------------------------------------------------------------------

@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("key", UNCATALOGUED_BEAMS, ids=_class)
@pytest.mark.parametrize("fmt", FORMATS)
def test_uncatalogued_beam_global_force_only_is_named(
    tmp_path: Path, fmt: str, key: str, comp: str,
) -> None:
    assert _class(key) not in LINE_STATION_NON_BEAM_CLASSES
    _assert_names(fmt, tmp_path / f"beam.{fmt}", key, comp)


@pytest.mark.parametrize("key", NAMED_NON_BEAMS, ids=_class)
@pytest.mark.parametrize("fmt", FORMATS)
def test_joint_and_cable_are_named_fail_closed(
    tmp_path: Path, fmt: str, key: str,
) -> None:
    _assert_names(fmt, tmp_path / f"joint.{fmt}", key, "bending_moment_z")


@pytest.mark.parametrize("fmt", FORMATS)
def test_unparseable_bracket_key_is_named(tmp_path: Path, fmt: str) -> None:
    """A key whose class does not parse is not known to be a non-beam."""
    key = "ElastomericBearingPlasticity3d"
    f = _File(fmt, tmp_path / f"unparsed.{fmt}")
    f.add("globalForce", [11], key=key)
    with f.reader() as r:
        with pytest.warns(SKIP_WARNING[fmt],
                          match=rf"globalForce/{key}.*\[11\].*never a"):
            slab = _read(r, "bending_moment_z", [11])
        assert slab.element_index.size == 0


# ---------------------------------------------------------------------
# A non-beam stays silent
# ---------------------------------------------------------------------

@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("key", NON_BEAMS, ids=_class)
@pytest.mark.parametrize("fmt", FORMATS)
def test_non_beam_global_force_is_silent(
    tmp_path: Path, fmt: str, key: str, comp: str,
) -> None:
    f = _File(fmt, tmp_path / f"nonbeam.{fmt}")
    f.add("globalForce", [11], key=key)
    with f.reader() as r:
        for ids in ([11], None):
            with warnings.catch_warnings():
                warnings.simplefilter("error", SKIP_WARNING[fmt])
                slab = _read(r, comp, ids)
            assert slab.element_index.size == 0


# ---------------------------------------------------------------------
# The deny-list is derived from the catalog's family tables
# ---------------------------------------------------------------------

def test_deny_list_is_the_union_of_the_family_tables() -> None:
    derived = (
        set(NON_BEAM_LINE_CLASSES)
        | {c for c, _rule, _tok in RESPONSE_CATALOG}
        | {c for c, _tok in LAYER_CATALOG}
        | {c for c, _tok in ZEROLENGTH_CATALOG}
    )
    assert LINE_STATION_NON_BEAM_CLASSES == derived


def test_deny_list_holds_no_catalogued_beam() -> None:
    beams = (
        {c for c, _tok in NODAL_FORCE_CATALOG}
        | {c for c, _kind in CUSTOM_RULE_CATALOG}
        | {c for c, _tok in FIBER_CATALOG}
    )
    assert not (LINE_STATION_NON_BEAM_CLASSES & beams)
