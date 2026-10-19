"""Replay keeps each argv slot's int/float type (K R1b-S2, #2194).

The bridge archives a primitive's positional arguments as one ``float64``
array (``_write_param_array``, the element argstack), which keeps no
int/float type. The readers recover a type by value: an integral float
comes back as an ``int``. That is right for a tag or a count and wrong for
a coefficient the bridge wrote as a float: ``thickness=1.0`` replays as
``1``, ``E=2e11`` as ``200000000000``.

``/opensees/decl_params`` (K1-7, schema 2.26.0) stores every declaration's
fields as JSON, so a float field stays a float. :class:`SlotTypes`
rebuilds the archived primitive from those fields, runs its ``_emit`` on a
:class:`~apeGmsh.opensees.emitter.recording.RecordingEmitter`, and takes
**only the int/float type** of each slot of the recorded call. The
archived values are never replaced: a slot whose dry-run value is a float
gets ``float(value)``, one whose dry-run value is an int keeps the reader's
int. Each ``$decl`` reference is a stub of the referenced class (an
instance with no fields set), resolved to a placeholder tag: a tag slot is
an int whatever its value.

Nothing changes, and the reader's by-value recovery stands, when the
archive has no ``decl_params`` (every file below 2.26.0, a rewrite of
one), when no declaration owns the tag, or when the recorded call's
argument count differs from the archived one. A declaration whose class
this build does not know, or whose rebuild or dry run raises, warns
:class:`ReplaySlotTypeWarning` once and keeps the by-value recovery: the
replay still runs, and the warning names the declaration.
"""
from __future__ import annotations

import bisect
import dataclasses
import sys
import warnings
from collections.abc import Mapping
from typing import Any, Sequence

from .. import damping as _damping  # noqa: F401  (registers the classes)
from .. import element as _element  # noqa: F401
from .. import integration as _integration  # noqa: F401
from .. import section as _section  # noqa: F401
from .. import time_series as _time_series  # noqa: F401
from .. import transform as _transform  # noqa: F401
from ..material import nd as _nd  # noqa: F401
from ..material import uniaxial as _uniaxial  # noqa: F401
from ..emitter.h5_reader import (
    DeclarationTable, DeclOpaque, DeclRef, DeclStruct,
)
from ..emitter.recording import RecordingEmitter
from .tag_resolution import set_element_nodes, set_tag_resolver
from .types import (
    BeamIntegration, Damping, Element, GeomTransf, NDMaterial, Pattern,
    Primitive, Recorder, Section, TimeSeries, UniaxialMaterial,
)

__all__ = ["ReplaySlotTypeWarning", "SlotTypes", "primitive_classes"]


class ReplaySlotTypeWarning(UserWarning):
    """A declaration's slot types could not be recovered from
    ``/opensees/decl_params``: its class is unknown to this build, or its
    rebuild or dry run raised. Its integral floats replay as ints."""


#: Each ``/opensees/decls`` family a primitive belongs to, and its base.
_FAMILY_BASE: dict[str, type[Primitive]] = {
    "uniaxialMaterial": UniaxialMaterial,
    "nDMaterial": NDMaterial,
    "section": Section,
    "geomTransf": GeomTransf,
    "beamIntegration": BeamIntegration,
    "element": Element,
    "timeSeries": TimeSeries,
    "pattern": Pattern,
    "recorder": Recorder,
    "damping": Damping,
}

#: The emitter verbs that write a store row of each tagged kind, as the
#: :class:`RecordingEmitter` names them.
_VERBS: dict[str, frozenset[str]] = {
    "uniaxialMaterial": frozenset({"uniaxialMaterial"}),
    "nDMaterial": frozenset({"nDMaterial"}),
    "section": frozenset({"section", "section_open"}),
    "beamIntegration": frozenset({"beamIntegration"}),
    "timeSeries": frozenset({"timeSeries"}),
    "damping": frozenset({"damping"}),
    "element": frozenset({"element"}),
}

#: The placeholder tag every stub reference resolves to.
_STUB_TAG = 1

_CLASSES: dict[str, dict[str, type[Primitive]]] | None = None


def _subclasses(cls: type) -> "list[type]":
    out: list[type] = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_subclasses(sub))
    return out


def primitive_classes() -> dict[str, dict[str, type[Primitive]]]:
    """``family -> class name -> class`` over every concrete primitive.

    A class counts once, as its module exports it (``slots=True``
    dataclasses leave their pre-slots class behind as a subclass). A name
    two classes of one family share is left out, so a lookup of it fails
    instead of picking one."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    found: dict[str, dict[str, set[type[Primitive]]]] = {
        fam: {} for fam in _FAMILY_BASE}
    for cls in _subclasses(Primitive):
        if getattr(cls, "__abstractmethods__", None):
            continue
        module = sys.modules.get(cls.__module__)
        if getattr(module, cls.__qualname__, None) is not cls:
            continue
        for fam, base in _FAMILY_BASE.items():
            if issubclass(cls, base):
                found[fam].setdefault(cls.__name__, set()).add(cls)
    _CLASSES = {
        fam: {name: next(iter(c)) for name, c in names.items() if len(c) == 1}
        for fam, names in found.items()
    }
    return _CLASSES


class _Unrecoverable(Exception):
    """The declaration cannot be rebuilt; the message says why."""


def _stub(cls: type[Primitive]) -> Primitive:
    """An instance of ``cls`` with no field set: it passes an
    ``isinstance`` check and reads its class attributes, and only a tag
    lookup may use it."""
    stub: Primitive = object.__new__(cls)
    return stub


class SlotTypes:
    """The int/float type of each archived argv slot, from the archive's
    ``/opensees/decl_params`` (see the module docstring).

    ``table`` is the archive's :meth:`H5Model.declarations`; ``None`` (an
    archive without ``decls``) recovers nothing. One instance serves one
    archive and caches each declaration's dry run.
    """

    def __init__(self, table: "DeclarationTable | None") -> None:
        self._table = table
        self._row_of_key: dict[str, int] = {}
        self._runs: dict[str, tuple[list[int], list[tuple[int, int]]]] = {}
        if table is not None and table.params:
            self._row_of_key = {d.key: i for i, d in enumerate(table.decls)}
            runs: dict[str, list[tuple[int, int, int]]] = {}
            for kind, first, count, decl in table.tags:
                runs.setdefault(kind, []).append((first, count, decl))
            for kind, rs in runs.items():
                rs.sort()
                self._runs[kind] = (
                    [r[0] for r in rs], [(r[1], r[2]) for r in rs])
        self._types: dict[
            tuple[int, str, int, int], "tuple[Any, ...] | None"] = {}

    @property
    def active(self) -> bool:
        """``True`` when the archive stores declaration parameters."""
        return bool(self._runs)

    def _decl_of(self, kind: str, tag: int) -> int | None:
        found = self._runs.get(kind)
        if found is None:
            return None
        firsts, rest = found
        i = bisect.bisect_right(firsts, tag) - 1
        if i < 0:
            return None
        count, decl = rest[i]
        return decl if tag < firsts[i] + count else None

    def recover(
        self, kind: str, tag: int, type_token: str,
        args: "Sequence[Any]", *, nodes: int = 0,
    ) -> "tuple[Any, ...]":
        """``args`` with each numeric slot cast to the type the archived
        primitive emits there.

        ``args`` is a store row after its type token and tag (for an
        element, after its ``nodes`` connectivity slots too). Returned
        unchanged when nothing is known (see the module docstring).
        """
        out = tuple(args)
        if not self._runs:
            return out
        decl = self._decl_of(kind, int(tag))
        if decl is None:
            return out
        key = (decl, type_token, int(tag) if kind != "element" else 0, nodes)
        if key not in self._types:
            self._types[key] = self._dry_run(
                decl, kind, int(tag), type_token, nodes)
        recorded = self._types[key]
        if recorded is None or len(recorded) != len(out):
            return out
        typed: list[Any] = []
        for got, want in zip(out, recorded):
            if (isinstance(got, (int, float)) and not isinstance(got, bool)
                    and isinstance(want, float)):
                typed.append(float(got))
            else:
                typed.append(got)
        return tuple(typed)

    # -- The dry run -------------------------------------------------------

    def _dry_run(
        self, decl: int, kind: str, tag: int, type_token: str, nodes: int,
    ) -> "tuple[Any, ...] | None":
        table = self._table
        assert table is not None
        entries = table.params.get(decl, ())
        if len(entries) != 1:
            return None
        where = f"{table.decls[decl].key!r} ({entries[0].type})"
        try:
            prim = self._rebuild(decl)
            rec = RecordingEmitter()
            set_tag_resolver(rec, lambda _p: _STUB_TAG)
            if kind == "element":
                set_element_nodes(rec, tuple(range(1, nodes + 1)))
            prim._emit(rec, tag)
        except Exception as exc:  # noqa: BLE001 - every cause warns, below
            warnings.warn(
                f"replay: the slot types of declaration {where} are not "
                f"recovered ({type(exc).__name__}: {exc}); its integral "
                "floats replay as ints.",
                ReplaySlotTypeWarning, stacklevel=2)
            return None
        verbs = _VERBS[kind]
        for name, call_args, _kw in rec.calls:
            if (name in verbs and len(call_args) >= 2
                    and call_args[0] == type_token and call_args[1] == tag):
                return tuple(call_args[2 + (nodes if kind == "element" else 0):])
        return None

    def _class(self, family: str, type_name: str) -> type[Primitive]:
        classes = primitive_classes().get(family)
        if classes is None:
            raise _Unrecoverable(
                f"the family {family!r} has no primitive classes")
        cls = classes.get(type_name)
        if cls is None:
            raise _Unrecoverable(
                f"this build has no {family} class {type_name!r}")
        return cls

    def _rebuild(self, decl: int) -> Primitive:
        """The primitive declaration ``decl`` archived, its references
        stubbed."""
        table = self._table
        assert table is not None
        (entry,) = table.params[decl]
        cls = self._class(table.decls[decl].family, entry.type)
        kwargs = self._fields(cls, entry.params, cls.__module__)
        return cls(**kwargs)

    def _fields(
        self, cls: type, params: Mapping[str, Any], module: str,
    ) -> dict[str, Any]:
        init = {f.name for f in dataclasses.fields(cls) if f.init}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise _Unrecoverable(
                f"{cls.__name__} has no field(s) {sorted(unknown)}")
        return {
            name: self._value(value, module)
            for name, value in params.items() if name in init}

    def _value(self, value: Any, module: str) -> Any:
        if isinstance(value, DeclRef):
            table = self._table
            assert table is not None
            row = self._row_of_key[value.key]
            entries = table.params.get(row, ())
            if len(entries) != 1:
                raise _Unrecoverable(
                    f"the reference {value.key!r} has {len(entries)} "
                    "parameter rows")
            return _stub(self._class(table.decls[row].family, entries[0].type))
        if isinstance(value, DeclStruct):
            struct = getattr(sys.modules.get(module), value.type, None)
            if not (isinstance(struct, type)
                    and dataclasses.is_dataclass(struct)):
                raise _Unrecoverable(
                    f"the value class {value.type!r} is not in {module}")
            return struct(**self._fields(struct, value.fields, module))
        if isinstance(value, DeclOpaque):
            raise _Unrecoverable(
                f"the field holds a {value.type}, archived by name only")
        if isinstance(value, tuple):
            return tuple(self._value(v, module) for v in value)
        if isinstance(value, Mapping):
            return {k: self._value(v, module) for k, v in value.items()}
        return value
