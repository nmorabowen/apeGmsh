"""Out-of-process studio contract (ADR 0095 INV-17 / S5d).

Published JSON Schema files live under ``apeGmsh/studio/schemas/``.
Golden fixtures under ``schemas/fixtures/`` are what an out-of-tree
consumer copies instead of importing the FEM stack.

``contract_version`` is a semver for the *habitat contract* (distinct
from per-file integer ``schema`` fields). Additive fields never bump
the major; readers ignore unknown keys; unknown major is refused.
Since 1.8.0 the two snapshots a consumer polls most — ``names.json``
and ``progress.json`` — carry the stamp themselves, so the major gate
works off a file on disk and not only off a live ``status`` call.

``runs.jsonl`` interleaves TWO record kinds and therefore has two
published schemas: a run line carries no ``kind`` and validates as
``"ledger"``; a pin line carries ``kind="pin"`` and validates as
``"ledger_pin"``. A consumer walking the file discriminates on
``kind`` first. They are separate schemas rather than one ``oneOf``
because the validator below is a deliberate draft-07 SUBSET (type /
required / properties / items / enum / const) that a Workbench-side
reimplementation can match without a JSON Schema library — and
relaxing the run schema's ``required`` list to admit both would stop
it catching a run line that lost its ``script``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1.8.0"
CONTRACT_MAJOR = 1

# kind → (schema file stem, expected per-file schema int)
_KINDS: dict[str, tuple[str, int]] = {
    "selection": ("selection", 1),
    "names": ("names", 1),
    "ledger": ("ledger", 1),
    "ledger_pin": ("ledger_pin", 1),
    "highlight": ("highlight", 1),
    "status": ("status", 1),
    "host": ("host", 1),
    "project": ("project", 1),
    "busy": ("busy", 1),
    "assess": ("assess", 1),
    "progress": ("progress", 1),
}


class ContractError(ValueError):
    """Payload does not satisfy the published studio contract."""


def parse_contract_version(version: str) -> tuple[int, int, int]:
    text = str(version).strip()
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        raise ContractError(f"invalid contract_version: {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def check_contract_version(version: str) -> None:
    """Refuse unknown major (INV-17). Same major is accepted."""
    major, _minor, _patch = parse_contract_version(version)
    if major != CONTRACT_MAJOR:
        raise ContractError(
            f"unsupported contract major {major} "
            f"(this library speaks {CONTRACT_VERSION})"
        )


def schemas_dir() -> Path:
    """Directory of published ``*.schema.json`` files."""
    return Path(__file__).resolve().parent / "schemas"


def fixtures_dir() -> Path:
    return schemas_dir() / "fixtures"


def load_schema(kind: str) -> dict[str, Any]:
    if kind not in _KINDS:
        raise ContractError(f"unknown contract kind: {kind!r}")
    stem, _ = _KINDS[kind]
    path = schemas_dir() / f"{stem}.schema.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"schema {path.name} must be an object")
    return data


def load_fixture(kind: str) -> dict[str, Any]:
    if kind not in _KINDS:
        raise ContractError(f"unknown contract kind: {kind!r}")
    stem, _ = _KINDS[kind]
    path = fixtures_dir() / f"{stem}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"fixture {path.name} must be an object")
    return data


def validate(kind: str, data: Any) -> None:
    """Validate *data* against the published schema for *kind*.

    Raises :class:`ContractError` on failure. Uses a draft-07 subset
    (type / required / properties / items / enum / const) so Workbench
    does not need the FEM stack or ``jsonschema`` to share the rules —
    the JSON Schema files remain the portable source of truth.
    """
    if kind not in _KINDS:
        raise ContractError(f"unknown contract kind: {kind!r}")
    if not isinstance(data, dict):
        raise ContractError(f"{kind} payload must be an object")
    _, expected_schema = _KINDS[kind]
    raw_schema = data.get("schema")
    if raw_schema is not None:
        try:
            got = int(raw_schema)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{kind}.schema must be an int") from exc
        if got != expected_schema:
            raise ContractError(
                f"{kind}.schema={got} unsupported "
                f"(expected {expected_schema})"
            )
    # Any payload MAY carry ``contract_version``; when it does, the major
    # gate applies (INV-17). ``status`` was the only carrier until 1.8.0
    # put the stamp on the ``names.json`` / ``progress.json`` files that
    # actually land on disk — a consumer that only ever reads files could
    # not otherwise engage the gate at all.
    ver = data.get("contract_version")
    if ver is not None:
        check_contract_version(str(ver))
    schema = load_schema(kind)
    _validate_against(schema, data, path=kind)


def _validate_against(schema: dict[str, Any], data: Any, *, path: str) -> None:
    if "const" in schema and data != schema["const"]:
        raise ContractError(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and data not in schema["enum"]:
        raise ContractError(f"{path}: value not in enum")
    expected = schema.get("type")
    if expected is not None:
        _check_type(expected, data, path=path)
    if isinstance(data, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in data:
                raise ContractError(f"{path}: missing required {key!r}")
        props = schema.get("properties") or {}
        for key, value in data.items():
            if key in props and isinstance(props[key], dict):
                _validate_against(props[key], value, path=f"{path}.{key}")
    if isinstance(data, (list, tuple)) and "items" in schema:
        item_schema = schema["items"]
        if isinstance(item_schema, dict):
            for i, item in enumerate(data):
                _validate_against(item_schema, item, path=f"{path}[{i}]")


def _check_type(expected: str | list[str], data: Any, *, path: str) -> None:
    kinds = expected if isinstance(expected, list) else [expected]
    ok = False
    for kind in kinds:
        if kind == "object" and isinstance(data, dict):
            ok = True
        elif kind == "array" and isinstance(data, (list, tuple)):
            ok = True
        elif kind == "string" and isinstance(data, str):
            ok = True
        elif kind == "integer" and isinstance(data, int) and not isinstance(data, bool):
            ok = True
        elif kind == "number" and isinstance(data, (int, float)) and not isinstance(
            data, bool
        ):
            ok = True
        elif kind == "boolean" and isinstance(data, bool):
            ok = True
        elif kind == "null" and data is None:
            ok = True
    if not ok:
        raise ContractError(
            f"{path}: expected type {expected!r}, got {type(data).__name__}"
        )
