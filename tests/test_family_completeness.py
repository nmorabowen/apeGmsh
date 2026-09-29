"""Every concrete public primitive is in a family and in its ``ALL_*`` list.

The fail-closed family-completeness gate (program C1.1; panel S3). It
imports every module under ``apeGmsh.opensees``, walks the ``Primitive``
subclass tree, and fails when a concrete, public class

* belongs to no family in ``tests/families.py``, or to more than one;
* is missing from its family's ``ALL_*`` contract list;

unless ``EXCEPTIONS`` lists it with a reason. A stale exception (the
class is now covered, or no longer exists) also fails, so the list only
shrinks. The self-tests at the bottom run the same checks on synthetic
classes, so "a class dropped from an ``ALL_*`` list turns the gate red"
is proven on every run, not only once in a PR body.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pytest

import apeGmsh.opensees
from apeGmsh.opensees._internal.types import Primitive
from tests.families import EXCEPTIONS, EXCEPTIONS_BASELINE, FAMILIES, Family

_SCOPE = "apeGmsh.opensees."
_ENGINE_MODULES = ("openseespy", "opensees")


def _key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _bound(cls: type) -> object:
    obj: object = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        obj = getattr(obj, part, None)
    return obj


def _import_all() -> None:
    def _fail(name: str) -> None:
        raise ImportError(f"walk could not import package {name!r}")

    for info in pkgutil.walk_packages(
        apeGmsh.opensees.__path__, _SCOPE, onerror=_fail
    ):
        importlib.import_module(info.name)


def walk() -> list[type[Primitive]]:
    """Every concrete, public ``Primitive`` subclass in ``apeGmsh.opensees``.

    ``@dataclass(slots=True)`` returns a new class and leaves its
    pre-image in ``__subclasses__``; that pre-image is skipped because its
    name is bound to the slotted twin. Any other concrete class whose name
    does not resolve to itself is an error, not a skip.
    """
    _import_all()
    found: dict[str, type[Primitive]] = {}
    unreachable: list[str] = []
    seen: set[type] = set()
    stack: list[type] = [Primitive]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
            if not sub.__module__.startswith(_SCOPE):
                continue
            if inspect.isabstract(sub) or sub.__name__.startswith("_"):
                continue
            bound = _bound(sub)
            if bound is sub:
                found[_key(sub)] = sub
            elif not (
                isinstance(bound, type)
                and _key(bound) == _key(sub)
                and "__slots__" in vars(bound)
            ):
                unreachable.append(_key(sub))
    assert not unreachable, (
        "concrete Primitive subclasses not reachable by module:qualname "
        f"(the gate cannot name them): {sorted(unreachable)}"
    )
    return sorted(found.values(), key=_key)


def resolve(contract: str) -> list[type[Primitive]]:
    module, _, name = contract.partition(":")
    return list(getattr(importlib.import_module(module), name))


def members(cls: type, families: Iterable[Family]) -> list[Family]:
    return [
        f for f in families
        if issubclass(cls, f.base) and (not f.modules or cls.__module__ in f.modules)
    ]


@dataclass(frozen=True)
class Gaps:
    unclassified: list[str]
    ambiguous: list[str]
    unlisted: list[str]
    stale: list[str]
    misfiled: list[str]

    def messages(self) -> list[str]:
        return [
            *(f"{k} is in no family; add a Family to tests/families.py or an "
              "EXCEPTIONS entry with a reason" for k in self.unclassified),
            *(f"{k} matches more than one family: {fams}" for k, fams in self.ambiguous),
            *(f"{k} ({fam}) is missing from {contract}; append it there"
              for k, fam, contract in self.unlisted),
            *(f"stale EXCEPTIONS entry {k}: {why}; delete it" for k, why in self.stale),
            *(f"{contract} lists {k}, which is not a {fam}" for k, fam, contract in self.misfiled),
        ]


def find_gaps(
    classes: Sequence[type],
    families: Sequence[Family],
    contracts: Mapping[str, Sequence[type]],
    exceptions: Mapping[str, str],
) -> Gaps:
    unclassified, ambiguous, unlisted, stale, misfiled = [], [], [], [], []
    walked = {_key(c) for c in classes}
    covered: set[str] = set()
    for cls in classes:
        key = _key(cls)
        fams = members(cls, families)
        if len(fams) > 1:
            ambiguous.append((key, [f.name for f in fams]))
        elif not fams:
            if key not in exceptions:
                unclassified.append(key)
        elif cls in contracts[fams[0].name]:
            covered.add(key)
        elif key not in exceptions:
            unlisted.append((key, fams[0].name, fams[0].contract))
    for key in exceptions:
        if key not in walked:
            stale.append((key, "no concrete public primitive by that name"))
        elif key in covered:
            stale.append((key, "the class is now in its family's ALL_* list"))
    for fam in families:
        for cls in contracts[fam.name]:
            if fam not in members(cls, families):
                misfiled.append((_key(cls), fam.name, fam.contract))
    return Gaps(unclassified, ambiguous, unlisted, stale, misfiled)


def ratchet(exceptions: Mapping[str, str], baseline: int) -> list[str]:
    if len(exceptions) <= baseline:
        return []
    return [
        f"EXCEPTIONS has {len(exceptions)} entries, above the baseline of "
        f"{baseline}; the ratchet only shrinks. Add the class to its ALL_* "
        "list instead (raising EXCEPTIONS_BASELINE needs the maintainer)"
    ]


# --- the gate ---------------------------------------------------------------


@pytest.fixture(scope="module")
def primitives() -> list[type[Primitive]]:
    before = {m for m in _ENGINE_MODULES if m in sys.modules}
    classes = walk()
    loaded = {m for m in _ENGINE_MODULES if m in sys.modules} - before
    assert not loaded, f"the walk imported an OpenSees engine: {sorted(loaded)}"
    return classes


@pytest.fixture(scope="module")
def contracts() -> dict[str, list[type[Primitive]]]:
    return {f.name: resolve(f.contract) for f in FAMILIES}


def test_every_family_has_members(primitives: list[type[Primitive]]) -> None:
    empty = [f.name for f in FAMILIES if not any(f in members(c, FAMILIES) for c in primitives)]
    assert not empty, f"families with no concrete primitive (a misspelt module?): {empty}"


def test_family_names_are_unique() -> None:
    names = [f.name for f in FAMILIES]
    assert len(names) == len(set(names))


def test_every_primitive_is_in_its_family_contract_list(
    primitives: list[type[Primitive]],
    contracts: dict[str, list[type[Primitive]]],
) -> None:
    gaps = find_gaps(primitives, FAMILIES, contracts, EXCEPTIONS)
    assert not gaps.messages(), "\n".join(gaps.messages())


def test_exceptions_never_grow_past_the_baseline() -> None:
    assert not ratchet(EXCEPTIONS, EXCEPTIONS_BASELINE), ratchet(EXCEPTIONS, EXCEPTIONS_BASELINE)[0]


def test_every_exception_has_a_reason() -> None:
    bare = [k for k, why in EXCEPTIONS.items() if len(why.strip()) < 20]
    assert not bare, f"EXCEPTIONS entries without a reason: {bare}"


# --- self-tests on synthetic classes (no Primitive subclass is created) -----


class _Base:
    pass


class _Other:
    pass


class _A(_Base):
    pass


class _B(_Base):
    pass


_FAKE = (Family("fake", _Base, "fake:ALL_FAKE"),)  # type: ignore[arg-type]


def _gaps(classes: list[type], listed: list[type], exceptions: dict[str, str]) -> list[str]:
    return find_gaps(classes, _FAKE, {"fake": listed}, exceptions).messages()


def test_self_passes_a_complete_family() -> None:
    assert _gaps([_A, _B], [_A, _B], {}) == []


def test_self_flags_a_class_dropped_from_its_list() -> None:
    assert _gaps([_A, _B], [_A], {}) == [
        f"{_key(_B)} (fake) is missing from fake:ALL_FAKE; append it there"
    ]


def test_self_excepted_gap_passes() -> None:
    assert _gaps([_A, _B], [_A], {_key(_B): "a reason"}) == []


def test_self_flags_a_class_in_no_family() -> None:
    [msg] = _gaps([_A, _Other], [_A], {})
    assert msg.startswith(f"{_key(_Other)} is in no family")


def test_self_flags_a_stale_exception_for_a_covered_class() -> None:
    [msg] = _gaps([_A], [_A], {_key(_A): "a reason"})
    assert msg.startswith(f"stale EXCEPTIONS entry {_key(_A)}: the class is now")


def test_self_flags_a_stale_exception_for_a_missing_class() -> None:
    [msg] = _gaps([_A], [_A], {"gone.module:Gone": "a reason"})
    assert msg.startswith("stale EXCEPTIONS entry gone.module:Gone: no concrete")


def test_self_flags_a_misfiled_list_entry() -> None:
    [msg] = _gaps([_A], [_A, _Other], {})
    assert msg == f"fake:ALL_FAKE lists {_key(_Other)}, which is not a fake"


def test_self_flags_ambiguous_membership() -> None:
    both = (*_FAKE, Family("also", _Base, "also:ALL"))  # type: ignore[arg-type]
    [msg] = find_gaps([_A], both, {"fake": [_A], "also": [_A]}, {}).messages()
    assert msg.startswith(f"{_key(_A)} matches more than one family")


def test_self_flags_an_exception_list_above_the_baseline() -> None:
    assert ratchet({"m:A": "a reason"}, 1) == []
    [msg] = ratchet({"m:A": "a reason", "m:B": "a reason"}, 1)
    assert msg.startswith("EXCEPTIONS has 2 entries, above the baseline of 1")
