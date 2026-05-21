"""ADR 0019 INV-3 / read-side-broker immutability acceptance tests.

* The :class:`OpenSeesModel` wrapper is frozen — assigning to any
  field raises ``FrozenInstanceError``.
* Record collections are read-only views (tuples / ``MappingProxyType``);
  ``.append`` / ``__setitem__`` raises.
* Every typed record is itself frozen.
* The module has no ``h5py`` write surface — AST scan mirrors the
  ``test_model_data_ast_guard`` precedent.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5


OPENSEES_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "apeGmsh" / "opensees" / "opensees_model.py"
)


# h5py write APIs.  Mirror ``test_model_data_ast_guard.py``.
FORBIDDEN_METHOD_NAMES = frozenset({
    "create_group",
    "create_dataset",
    "create_virtual_dataset",
    "require_group",
    "require_dataset",
    "move",
    "copy",
})


# ---------------------------------------------------------------------------
# Wrapper-level frozen
# ---------------------------------------------------------------------------

def test_cannot_assign_to_fem(tmp_path: Path) -> None:
    """Reassigning the FEM attribute on the frozen wrapper raises."""
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        om._fem = object()  # type: ignore[misc]
    # ``fem`` is a property with no setter — frozen-dataclass-with-slots
    # raises ``TypeError`` from the generated ``__setattr__``; bare
    # frozen raises ``FrozenInstanceError`` / ``AttributeError``.
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        om.fem = object()  # type: ignore[misc]


def test_cannot_assign_to_snapshot_id(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        om._snapshot_id = "abc"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Record collections — read-only views
# ---------------------------------------------------------------------------

def test_materials_collection_is_read_only(tmp_path: Path) -> None:
    """``om.materials()`` returns a tuple — has no ``.append``."""
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    mats = om.materials()
    assert isinstance(mats, tuple)
    with pytest.raises(AttributeError):
        mats.append(object())  # type: ignore[attr-defined]


def test_materials_by_family_is_read_only(tmp_path: Path) -> None:
    """``om.materials_by_family()`` is a ``MappingProxyType`` —
    assignment raises ``TypeError``."""
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    mapping = om.materials_by_family()
    from types import MappingProxyType
    assert isinstance(mapping, MappingProxyType)
    with pytest.raises(TypeError):
        mapping["uniaxial"] = ()  # type: ignore[index]


def test_sections_transforms_recorders_are_tuples(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    for accessor in (
        om.sections,
        om.transforms,
        om.beam_integration,
        om.time_series,
        om.patterns,
        om.recorders,
        om.elements,
        om.fixes,
        om.masses,
        om.cuts,
        om.sweeps,
    ):
        result = accessor()
        assert isinstance(result, tuple), (
            f"{accessor.__name__}() must return a tuple; got "
            f"{type(result).__name__}"
        )


def test_record_collection_returns_stable_view(tmp_path: Path) -> None:
    """Repeated accessor calls return tuples that compare equal —
    the cached representation does not change between calls.
    """
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    a = om.materials()
    b = om.materials()
    assert a == b
    assert om.transforms() == om.transforms()


# ---------------------------------------------------------------------------
# Per-record frozenness
# ---------------------------------------------------------------------------

def test_cannot_mutate_material_record(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    mat = om.materials()[0]
    with pytest.raises(FrozenInstanceError):
        mat.tag = 99  # type: ignore[misc]


def test_cannot_mutate_transform_record(tmp_path: Path) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(src)
    t = om.transforms()[0]
    with pytest.raises(FrozenInstanceError):
        t.vec = (0.0, 0.0, 0.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# No h5py write surface — AST scan (INV-3)
# ---------------------------------------------------------------------------

def test_opensees_model_path_exists() -> None:
    """Sanity check — the file we walk has to be present."""
    assert OPENSEES_MODEL_PATH.is_file(), (
        f"opensees_model.py not found at {OPENSEES_MODEL_PATH}. If the "
        f"module has moved, update the path constant in this test."
    )


def test_no_h5py_write_surface() -> None:
    """ADR 0019 INV-3: ``OpenSeesModel`` holds no h5py write surface.

    Schema authority remains with :class:`H5Emitter` /
    ``_compose_model_h5``.  AST-scan ``opensees_model.py`` and forbid
    any ``.create_group(...)`` / ``.create_dataset(...)`` / ...
    call, any ``something.attrs[...] = ...`` assignment, and any
    ``h5py.File(<path>, mode!=r)`` call.
    """
    src = OPENSEES_MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(OPENSEES_MODEL_PATH))
    offences: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # 1. Forbidden write-API method calls.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_METHOD_NAMES:
                offences.append(
                    (node.lineno,
                     f"forbidden h5py write call: .{node.func.attr}(...)")
                )

        # 2. Assignment to ``something.attrs[...]``.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "attrs"
                ):
                    offences.append(
                        (node.lineno,
                         "forbidden write to .attrs[...]")
                    )

        # 3. ``h5py.File(...)`` with a write-capable mode.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "File"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "h5py"
        ):
            mode = _resolve_h5py_file_mode(node)
            if mode is not None and mode not in {"r"}:
                offences.append(
                    (node.lineno,
                     f"forbidden h5py.File mode={mode!r} "
                     f"(only 'r' is allowed)")
                )

    assert not offences, (
        "OpenSeesModel module must have no h5py write surface "
        "(ADR 0019 INV-3 — schema authority stays in H5Emitter / "
        "_compose_model_h5). Offences:\n"
        + "\n".join(
            f"  {OPENSEES_MODEL_PATH.name}:{ln}  {why}"
            for ln, why in offences
        )
    )


def _resolve_h5py_file_mode(call: ast.Call) -> "str | None":
    """Resolve the ``mode=`` arg of ``h5py.File(path, mode='r')``."""
    if len(call.args) >= 2:
        arg = call.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return None
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                return v
    return "r"
