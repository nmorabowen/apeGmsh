"""Bridge-handoff snippets for section documents (ADR 0080 B7).

:func:`handoff_snippet` renders the few lines that put a finished
section into a **frame model script** — the last mile the builder GUI's
"Copy handoff" button covers. It is deliberately smaller than
``export_script`` (ADR 0080 B4): the export reproduces the whole
authoring session standalone, while a snippet assumes an ``apeSees``
bridge named ``ops`` already exists and just registers the section on
it.

The two lanes hand off differently, and the difference is the ADR 0078
declaration/resolution split:

* **fiber** — the section IS the document, so the snippet constructs it
  literally: one ``ops.uniaxialMaterial.<Type>`` per used material, then
  ``ops.section.Fiber(...)``. Templates are expanded to literals with a
  provenance comment, exactly as ``export_script`` does (both render
  through the same helpers, so a snippet and an export never disagree).
* **continuum** — the section is a *mesh* the analyzer solves, which no
  snippet can carry. The snippet re-opens the ``.section.json`` and
  lowers it late: ``ComputedSection(analysis=doc.build())`` for the
  elastic lowering, or ``doc.to_section(ops)`` when the document carries
  a ``bars`` overlay (the ADR 0080 B3 fiber lowering). Numbers are never
  hand-copied out of the GUI — the document stays the source of truth.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._script_export import (
    FIBER_IMPORT_LINES,
    _fmt,
    _ident,
    fiber_collection_lines,
    fiber_material_lines,
    fiber_template_comments,
)

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from ._document import SectionDocument


__all__ = ["handoff_snippet"]


def handoff_snippet(
    doc: "SectionDocument", *, path: "str | Path | None" = None,
) -> str:
    """Render the paste-ready bridge handoff for ``doc``.

    Parameters
    ----------
    doc
        The section document, either lane.
    path
        Where the document lives on disk. **Continuum lane only** — the
        snippet re-opens the document, so it needs the path; a
        placeholder derived from the document name is emitted when this
        is omitted (the GUI passes the file it has open). Ignored by the
        fiber lane, which inlines everything.

    Returns
    -------
    str
        Python source assuming an ``apeSees`` bridge bound to ``ops``.
        Always ``compile()``-able.
    """
    if doc.kind == "fiber":
        return _fiber_snippet(doc)
    return _continuum_snippet(doc, path=path)


def _var(doc: "SectionDocument") -> str:
    """A valid identifier for the section, from the document name."""
    ident = _ident(doc.name or "section")
    return ident if not ident[:1].isdigit() else f"s_{ident}"


def _header(doc: "SectionDocument") -> str:
    return (
        f"# {doc.name or 'section'} — apeGmsh SectionDocument "
        f"({doc.kind} lane), section_doc_version "
        f"{doc.to_dict()['section_doc_version']}"
    )


def _fiber_snippet(doc: "SectionDocument") -> str:
    recipe = doc.build()
    var = _var(doc)
    out: list[str] = [_header(doc), *FIBER_IMPORT_LINES, ""]
    out += fiber_material_lines(doc, indent="")
    out.append("")
    out += fiber_template_comments(doc, indent="")
    out += fiber_collection_lines(recipe, indent="")
    gj = "None" if recipe.GJ is None else _fmt(recipe.GJ)
    out += [
        f"{var} = ops.section.Fiber(",
        "    patches=patches, layers=layers, fibers=points,",
        f"    GJ={gj}, name={_fmt(doc.name or 'section')},",
        ")",
        "",
    ]
    return "\n".join(out)


def _continuum_snippet(
    doc: "SectionDocument", *, path: "str | Path | None",
) -> str:
    data: dict[str, Any] = doc.to_dict()
    var = _var(doc)
    doc_path = (
        str(path) if path is not None
        else f"{_ident(doc.name or 'section')}.section.json"
    )
    out = [
        _header(doc),
        "from apeGmsh.sections import SectionDocument",
        "",
        f"doc = SectionDocument.open({_fmt(doc_path)})",
    ]
    if data.get("bars"):
        n_bars = len(data["bars"])
        out += [
            f"# {n_bars} bars entr{'y' if n_bars == 1 else 'ies'} — the "
            f"ADR 0080 B3 fiber lowering appends them to the Gauss fibers",
            f"{var} = doc.to_section(ops, name={_fmt(doc.name or 'section')})",
            "",
            "# elastic lowering instead (no bars):",
            f"#   {var} = ops.section.ComputedSection(analysis=doc.build())",
        ]
    else:
        out += [
            f"{var} = ops.section.ComputedSection(",
            "    analysis=doc.build(),",
            f"    name={_fmt(doc.name or 'section')},",
            ")",
            "",
            "# Gauss-fiber lowering instead (per-region uniaxial "
            "materials required):",
            f"#   {var} = doc.to_section(ops)",
        ]
    out.append("")
    return "\n".join(out)
