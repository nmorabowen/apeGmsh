"""ADR 0018 C5 — ``ModelData`` has no HDF5 write surface.

INV-1 / INV-3 acceptance test: structural assertion that
``src/apeGmsh/opensees/model_data.py`` neither calls any h5py
write API nor mutates ``.attrs[...]`` nor opens a file in a
write-capable mode.  Mirrors the
``test_viewers_pure_h5_consumer.py`` AST-walk precedent (ADR 0014).

Schema authority for the on-disk file stays in exactly one module
(``H5Emitter`` for ``/opensees/...`` and ``mesh/_femdata_h5_io.py``
for the neutral zone); ``ModelData`` is a fact-collector that
delegates serialization to those owners via the shared
``_compose_model_h5`` composer.  This test enforces that
delegation at the source level so a future drift gets caught at
PR review, not by a silent file-shape regression.
"""
from __future__ import annotations

import ast
from pathlib import Path


MODEL_DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "apeGmsh" / "opensees" / "model_data.py"
)


# h5py write APIs.  Any attribute call with one of these names is a
# write surface and forbidden in ``model_data.py``.  Read APIs
# (``get`` is intentionally NOT here — though we forbid it elsewhere
# per the optional-child probe hazard, that's INV-15's domain, not
# this test).
FORBIDDEN_METHOD_NAMES = frozenset({
    "create_group",
    "create_dataset",
    "create_virtual_dataset",
    "require_group",
    "require_dataset",
    "move",
    "copy",
})


def test_model_data_path_exists() -> None:
    """Sanity check — the file we walk has to be present."""
    assert MODEL_DATA_PATH.is_file(), (
        f"model_data.py not found at {MODEL_DATA_PATH}. If the module "
        f"has moved, update the path constant in this test."
    )


def _collect_offences(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, reason)`` for every write-surface offence."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    offences: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # 1. Forbidden method calls (`.create_group(...)`, etc.).
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_METHOD_NAMES:
                offences.append(
                    (node.lineno,
                     f"forbidden h5py write call: .{node.func.attr}(...)")
                )

        # 2. Assignment to ``something.attrs[<key>]``.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "attrs"
                ):
                    offences.append(
                        (node.lineno,
                         "forbidden write to .attrs[...] (attribute mutation)")
                    )

        # 3. ``h5py.File(<path>, mode)`` with a write-capable mode.
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
                     f"(only read mode 'r' is allowed)")
                )

    return offences


def _resolve_h5py_file_mode(call: ast.Call) -> "str | None":
    """Resolve the ``mode=`` argument of ``h5py.File(path, mode='r')``.

    Returns the literal mode string when statically inferable, or
    ``None`` when the call uses a non-literal expression (in which
    case the test cannot decide and conservatively allows it — but
    such a pattern is itself a smell worth a manual review).
    """
    # Positional second arg.
    if len(call.args) >= 2:
        arg = call.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return None
    # Keyword mode=.
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                return v
    # No explicit mode → default "r" (h5py default).
    return "r"


def test_model_data_has_no_h5py_write_surface() -> None:
    offences = _collect_offences(MODEL_DATA_PATH)
    assert not offences, (
        "ModelData module must have no h5py write surface (ADR 0018 "
        "INV-1 / INV-3 — schema authority stays in H5Emitter and "
        "mesh/_femdata_h5_io.py).  Offences:\n"
        + "\n".join(
            f"  {MODEL_DATA_PATH.name}:{ln}  {why}" for ln, why in offences
        )
    )
