"""Lock: every concrete element primitive resolves in the capability registry.

Program slice C1.3 (#1228).  ``element_capability(class_name)`` is the
single registry lookup; it answers the :data:`Unknown` sentinel for any
class name :data:`_ELEM_REGISTRY` cannot resolve after
:data:`_CLASS_TOKEN_ALIASES`.  Before it, every helper answered
``None``/``False`` for an unheard-of class and every caller read that as
"permissive, skip".

The lock enumerates every concrete ``Element`` subclass the bridge can
emit (every non-abstract subclass defined under
``apeGmsh.opensees.element``) and fails when its lookup is ``Unknown``,
unless the class sits on :data:`EXTRAS_ONLY` with a reason.  That list
is a ratchet: it can only shrink.  An entry whose class has gained a
registry entry, or no longer exists, fails as stale.

Negative proof (PR body of #1228): deleting the ``"stdBrick"`` registry
entry turns ``test_every_concrete_element_resolves`` red naming
``stdBrick``; removing the ``"ZeroLength"`` exception turns it red naming
``ZeroLength``; adding a spurious exception fails ``test_exception_list_is_not_stale``.

Pure unit test: no gmsh, no openseespy.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import pickle
import pkgutil

import pytest

import apeGmsh.opensees.element as _element_pkg
from apeGmsh.opensees._element_capabilities import (
    _ELEM_REGISTRY,
    _EXTRA_CLASS_NDF_OK,
    _EXTRA_CLASS_REQUIRED_FLOOR,
    Unknown,
    element_capability,
    element_class_ndf_ok,
    element_required_floor,
)
from apeGmsh.opensees._internal.types import Element

# ---------------------------------------------------------------------------
# The ratchet.  class name -> why it has no ``_ElemSpec`` today.
#
# Every entry is carried by ``_EXTRA_CLASS_NDF_OK`` / ``_EXTRA_CLASS_REQUIRED_FLOOR``
# instead (the module docstring on those tables is the audit), so the
# ``None``-returning helpers still classify them; only the single lookup
# says ``Unknown``.  Migrating one to an ``_ElemSpec`` means deleting its
# row here — ``test_exception_list_is_not_stale`` enforces that.  Adding
# a row needs a reason a reviewer can check against the element source.
# ---------------------------------------------------------------------------
EXTRAS_ONLY: dict[str, str] = {
    "forceBeamColumn": (
        "beamIntegration-based: no scalar slot tuple, special-cased via "
        "_FORCE_DISP_BEAMS; ndf carried by _EXTRA_CLASS_NDF_OK"
    ),
    "dispBeamColumn": (
        "beamIntegration-based: no scalar slot tuple, special-cased via "
        "_FORCE_DISP_BEAMS; ndf carried by _EXTRA_CLASS_NDF_OK"
    ),
    "LadrunoDispBeamColumn": (
        "Ladruno fork beam, ndm-dispatched like dispBeamColumn; ndf carried "
        "by _EXTRA_CLASS_NDF_OK"
    ),
    "LadrunoIMKBeam": (
        "Ladruno fork beam, ndm-dispatched like dispBeamColumn; ndf carried "
        "by _EXTRA_CLASS_NDF_OK"
    ),
    "InertiaTruss": (
        "no material or section dependency (mass only), no slot grammar; "
        "ndf carried by _EXTRA_CLASS_NDF_OK"
    ),
    "ASDShellT3": (
        "shell without a gmsh_etypes/slot entry; single-valued ndf {6} "
        "carried by _EXTRA_CLASS_NDF_OK"
    ),
    "ZeroLength": (
        "explicit-node, adaptive (any node ndf 1..6, ADR 0049); no mesh "
        "fan-out so no _ElemSpec; carried by _EXTRA_CLASS_NDF_OK"
    ),
    "ZeroLengthSection": (
        "explicit-node, NOT adaptive (demands 3/6 dof or OpenSees silently "
        "drops it); carried by _EXTRA_CLASS_NDF_OK"
    ),
    "CoupledZeroLength": (
        "explicit-node, adaptive like ZeroLength; carried by "
        "_EXTRA_CLASS_NDF_OK"
    ),
    "TwoNodeLink": (
        "explicit-node, adaptive like ZeroLength; carried by "
        "_EXTRA_CLASS_NDF_OK"
    ),
}


#: The ratchet's ceiling: the names EXTRAS_ONLY may hold.  Frozen at the
#: ten classes of 2026-09-29 (#1228).  A row may leave EXTRAS_ONLY when its
#: class gains an ``_ElemSpec``; a name may NOT join it.  Adding a name here
#: is a maintainer-gated baseline raise, never part of an element PR: the
#: point of the lock is that a new element cannot land as "extras-only"
#: by adding one row to ``_EXTRA_CLASS_NDF_OK`` and one row here.
EXTRAS_ONLY_BASELINE: frozenset[str] = frozenset({
    "forceBeamColumn",
    "dispBeamColumn",
    "LadrunoDispBeamColumn",
    "LadrunoIMKBeam",
    "InertiaTruss",
    "ASDShellT3",
    "ZeroLength",
    "ZeroLengthSection",
    "CoupledZeroLength",
    "TwoNodeLink",
})


def _ratchet_violations(exceptions: dict[str, str]) -> list[str]:
    """Names in *exceptions* that the frozen baseline does not allow."""
    return sorted(set(exceptions) - EXTRAS_ONLY_BASELINE)


def _concrete_element_classes() -> dict[str, type[Element]]:
    """Every non-abstract ``Element`` subclass defined under
    ``apeGmsh.opensees.element`` (the bridge's emitting primitives), keyed
    by class name — the key ``_internal/build.py`` looks up
    (``type(spec).__name__``)."""
    found: dict[str, type[Element]] = {}
    for info in pkgutil.walk_packages(
        _element_pkg.__path__, _element_pkg.__name__ + ".",
    ):
        mod = importlib.import_module(info.name)
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(cls, Element)
                and cls is not Element
                and cls.__module__ == mod.__name__
                and not inspect.isabstract(cls)
            ):
                found[name] = cls
    return found


CONCRETE: dict[str, type[Element]] = _concrete_element_classes()


def test_enumeration_is_not_empty() -> None:
    """Guard the lock against an import-path slip that enumerates nothing."""
    assert {"stdBrick", "ShellMITC4", "elasticBeamColumn", "ZeroLength"} <= set(CONCRETE)


@pytest.mark.parametrize("class_name", sorted(CONCRETE))
def test_every_concrete_element_resolves(class_name: str) -> None:
    spec = element_capability(class_name)
    if spec is Unknown:
        assert class_name in EXTRAS_ONLY, (
            f"element_capability({class_name!r}) is Unknown: add an "
            f"_ELEM_REGISTRY entry (or a _CLASS_TOKEN_ALIASES row) in "
            f"src/apeGmsh/opensees/_element_capabilities.py, or add "
            f"{class_name!r} to EXTRAS_ONLY here with a reason."
        )
        # The reason claims the extras tables carry it: hold that claim.
        assert element_class_ndf_ok(class_name) is not None, (
            f"{class_name!r} is on EXTRAS_ONLY but _EXTRA_CLASS_NDF_OK does "
            f"not carry it — it is unclassifiable everywhere."
        )
    else:
        assert any(spec is v for v in _ELEM_REGISTRY.values())


@pytest.mark.parametrize("class_name", sorted(EXTRAS_ONLY))
def test_exception_list_is_not_stale(class_name: str) -> None:
    assert class_name in CONCRETE, (
        f"EXTRAS_ONLY names {class_name!r}, which is no longer a concrete "
        f"element primitive — delete the row."
    )
    assert element_capability(class_name) is Unknown, (
        f"{class_name!r} now resolves in _ELEM_REGISTRY — delete its "
        f"EXTRAS_ONLY row (the list only shrinks)."
    )
    assert EXTRAS_ONLY[class_name].strip(), f"{class_name!r} needs a reason."


def test_exception_list_is_within_the_frozen_baseline() -> None:
    """EXTRAS_ONLY is shrink-only: no name outside EXTRAS_ONLY_BASELINE."""
    assert _ratchet_violations(EXTRAS_ONLY) == [], (
        "EXTRAS_ONLY grew past its frozen baseline. Give the class an "
        "_ELEM_REGISTRY entry instead; raising EXTRAS_ONLY_BASELINE is a "
        "maintainer-gated baseline raise."
    )
    assert len(EXTRAS_ONLY_BASELINE) == 10


def test_ratchet_self_test_rejects_a_new_exception() -> None:
    """The check that guards the baseline must fail on a grown list, so a
    new element + a new _EXTRA_CLASS_NDF_OK row + a new EXTRAS_ONLY row
    cannot stay green."""
    grown = dict(EXTRAS_ONLY, NewFancyElement="carried by extras (not allowed)")
    assert _ratchet_violations(grown) == ["NewFancyElement"]
    shrunk = {k: v for k, v in EXTRAS_ONLY.items() if k != "ZeroLength"}
    assert _ratchet_violations(shrunk) == []


def test_extras_tables_carry_only_exceptions() -> None:
    """The extras tables are the exception mechanism; a class registered in
    both places has two sources of truth."""
    registered = {k for k in _EXTRA_CLASS_NDF_OK if element_capability(k) is not Unknown}
    assert not registered, f"in _ELEM_REGISTRY and _EXTRA_CLASS_NDF_OK: {sorted(registered)}"
    assert set(_EXTRA_CLASS_NDF_OK) == set(EXTRAS_ONLY)
    assert set(_EXTRA_CLASS_REQUIRED_FLOOR) <= set(EXTRAS_ONLY)


# ---------------------------------------------------------------------------
# The sentinel itself
# ---------------------------------------------------------------------------

def test_unknown_for_unregistered_class() -> None:
    assert element_capability("NoSuchElement") is Unknown
    assert element_capability("") is Unknown


def test_alias_resolves_to_the_token_spec() -> None:
    assert element_capability("FourNodeQuad") is _ELEM_REGISTRY["quad"]
    assert element_capability("quad") is _ELEM_REGISTRY["quad"]
    assert element_capability("Truss") is _ELEM_REGISTRY["truss"]


def test_unknown_has_no_truth_value() -> None:
    """``Unknown`` cannot be read as the permissive ``None``/``False``."""
    with pytest.raises(TypeError, match="spec is Unknown"):
        bool(Unknown)
    with pytest.raises(TypeError):
        if not element_capability("NoSuchElement"):  # pragma: no cover
            pass
    assert Unknown is not None
    assert Unknown is not False
    assert repr(Unknown) == "Unknown"


def test_unknown_survives_copy_and_pickle_as_the_same_object() -> None:
    """An Enum member: ``is`` keeps working across copy/pickle boundaries."""
    assert copy.copy(Unknown) is Unknown
    assert copy.deepcopy(Unknown) is Unknown
    assert pickle.loads(pickle.dumps(Unknown)) is Unknown


def test_legacy_helpers_keep_their_none_contract() -> None:
    """The per-helper ``None`` answers are unchanged (callers outside this
    seam still depend on them); only the single lookup fails closed."""
    assert element_class_ndf_ok("NoSuchElement") is None
    assert element_required_floor("NoSuchElement", 3) is None
    assert element_class_ndf_ok("ZeroLength") == frozenset({1, 2, 3, 4, 5, 6})
