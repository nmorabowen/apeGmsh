"""Studio assess / animate workers (ADR 0095 S4b / S4d).

Loaded only by ``python -m apeGmsh.studio --assess`` / ``--animate``.
The MCP process shells here so it never imports Gmsh, Qt, or viewers
(INV-5). ``render`` is the existing ``python -m apeGmsh.viewers render``
door (ADR 0094 S5) — not duplicated.

Visors land under ``.apegmsh/visors/`` (generated; INV-12). MCP never
forwards ``setup=``; the ``kind=yield`` worker may use it internally
to attach a von Mises contour on auto-scaled deform (INV-11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._paths import visors_path

ANIMATE_KINDS = ("history", "yield", "formation")
ANIMATE_SHIPPED = ("history", "yield")
# Parity with apeGmsh.viewers.render stills: largest |u| → this fraction
# of the model diagonal. MCP never forwards setup=; kind=yield uses
# setup internally for a von Mises contour (INV-11).
_DEFORM_FRACTION = 0.12
_YIELD_COMPONENTS = (
    "von_mises_stress",
    "von_mises_shell",
    "fluency",
    "von_mises",
)


def assess_artifact(
    path: str | Path,
    *,
    figures: bool = False,
    model_h5: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """``fem.assess()`` or ``results.assess()`` on an on-disk artifact."""
    src = Path(path)
    if not src.is_file():
        return {
            "ok": False,
            "returncode": 2,
            "error": f"file not found: {src}",
            "path": str(src),
        }
    dest = Path(out_dir) if out_dir is not None else visors_path()
    model = Path(model_h5) if model_h5 is not None else None
    if _is_model_only(src):
        report = _open_fem(src).assess(figures=figures, out_dir=dest)
        source = "fem"
    else:
        report = _open_results(src, model).assess(
            figures=figures, out_dir=dest,
        )
        source = "results"
    return {
        "ok": True,
        "returncode": 0,
        "source": source,
        "path": str(src.resolve()),
        "text": report.text,
        "findings": [_finding_dict(f) for f in report.findings],
        "skipped": [
            {"code": code, "reason": reason} for code, reason in report.skipped
        ],
        "figures": [str(p) for p in report.figures],
    }


def animate_artifact(
    path: str | Path,
    output: str | Path,
    *,
    kind: str = "history",
    model_h5: str | Path | None = None,
    fps: int = 30,
    step_stride: int = 1,
) -> dict[str, Any]:
    """``results.export_animation`` for shipped ``kind=`` tokens.

    MCP never forwards ``setup=``. ``kind=yield`` is a von Mises
    contour on an auto-scaled deforming mesh (stills 0.12 of diagonal),
    not an iso-clip of ``f(σ)=fy`` and not a π-plane.
    """
    if kind not in ANIMATE_KINDS:
        raise ValueError(
            f"kind must be one of {ANIMATE_KINDS}; got {kind!r}"
        )
    if kind not in ANIMATE_SHIPPED:
        raise ValueError(
            f"animate kind={kind!r} is not shipped "
            "(formation is later; S4d ships history and yield)"
        )
    src = Path(path)
    if not src.is_file():
        return {
            "ok": False,
            "returncode": 2,
            "kind": kind,
            "error": f"file not found: {src}",
            "path": str(src),
        }
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model = Path(model_h5) if model_h5 is not None else None
    results = _open_results(src, model)
    extra: dict[str, Any] = {}
    if kind == "yield":
        component = resolve_fluency_component(results)
        scale = auto_deform_scale(results)
        extra["component"] = component
        extra["deform_scale"] = scale
        extra["picture"] = (
            "von Mises contour on auto-scaled deforming mesh "
            "(not an iso-clip of f(σ)=fy)"
        )
        written = results.export_animation(
            dest,
            fps=fps,
            step_stride=step_stride,
            deform=("displacement", scale),
            setup=_yield_setup(component),
        )
    else:
        written = results.export_animation(
            dest, fps=fps, step_stride=step_stride,
        )
    out = str(written) if written is not None else None
    payload = {
        "ok": True,
        "returncode": 0,
        "kind": kind,
        "path": str(src.resolve()),
        "output": out,
        "skipped": written is None,
    }
    payload.update(extra)
    return payload


def resolve_fluency_component(results: Any) -> str:
    """Pick von Mises / fluency from recorded + derived scalars (ADR 0076)."""
    from apeGmsh.results._derived import available_derived

    have = _recorded_components(results)
    have.update(available_derived(have))
    for name in _YIELD_COMPONENTS:
        if name in have:
            return name
    raise ValueError(
        "animate kind=yield needs von_mises_stress (or fluency) from a "
        "recorded stress tensor (ADR 0076). π-plane is later."
    )


def _recorded_components(results: Any) -> set[str]:
    inspect = getattr(results, "inspect", None)
    if inspect is None or not hasattr(inspect, "components"):
        return set()
    names: set[str] = set()
    try:
        by_level = inspect.components()
    except Exception:
        by_level = None
    if isinstance(by_level, dict):
        for comps in by_level.values():
            names.update(comps)
        return names
    for stage in getattr(results, "stages", ()) or ():
        sid = getattr(stage, "id", None)
        if sid is None:
            continue
        try:
            chunk = inspect.components(stage=sid)
        except Exception:
            continue
        if isinstance(chunk, dict):
            for comps in chunk.values():
                names.update(comps)
    return names


def auto_deform_scale(results: Any) -> float:
    """Stills-style warp: largest |u| becomes 0.12 of the model diagonal."""
    diag = _model_diagonal(results)
    max_d = _max_displacement(results)
    if max_d > 0.0 and diag > 0.0:
        return float(_DEFORM_FRACTION * diag / max_d)
    return 1.0


def _model_diagonal(results: Any) -> float:
    import numpy as np

    fem = getattr(results, "fem", None)
    if fem is None:
        return 0.0
    diag = getattr(fem, "model_diagonal", None)
    try:
        if diag is not None and float(diag) > 0.0:
            return float(diag)
    except (TypeError, ValueError):
        pass
    nodes = getattr(fem, "nodes", None)
    coords = getattr(nodes, "coords", None) if nodes is not None else None
    if coords is None:
        return 0.0
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        return 0.0
    span = arr.max(axis=0) - arr.min(axis=0)
    return float(np.linalg.norm(span))


def _max_displacement(results: Any) -> float:
    import numpy as np

    nodes = getattr(results, "nodes", None)
    if nodes is None or not hasattr(nodes, "get"):
        return 0.0
    comps: list[Any] = []
    for axis in ("x", "y", "z"):
        try:
            slab = nodes.get(component=f"displacement_{axis}", time=-1)
        except Exception:
            continue
        vals = getattr(slab, "values", slab)
        try:
            comps.append(np.asarray(vals, dtype=float))
        except (TypeError, ValueError):
            continue
    if not comps:
        return 0.0
    stacked = np.stack(comps, axis=-1)
    return float(np.linalg.norm(stacked, axis=-1).max())


def _yield_setup(component: str):
    """Internal ``export_animation(setup=)`` — not an MCP argument (INV-11)."""

    def setup(_plotter: Any, director: Any) -> None:
        from apeGmsh.viewers.diagrams._base import DiagramSpec
        from apeGmsh.viewers.diagrams._kinds import kind_def
        from apeGmsh.viewers.diagrams._selectors import (
            normalize as normalize_selector,
        )

        kdef = kind_def("contour")
        spec = DiagramSpec(
            kind="contour",
            selector=normalize_selector(component=component),
            style=kdef.make_default_style(component),
        )
        director.registry.add(kdef.diagram_class(spec, director.results))

    return setup


def dumps_assess(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _finding_dict(finding: Any) -> dict[str, Any]:
    detail = getattr(finding, "detail", None)
    return {
        "code": finding.code,
        "severity": finding.severity,
        "message": finding.message,
        "detail": _jsonable(detail),
    }


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _is_model_only(path: Path) -> bool:
    from apeGmsh.viewers.__main__ import _is_model_only as impl

    return impl(path)


def _open_fem(path: Path) -> Any:
    from apeGmsh.viewers.__main__ import _open_fem as impl

    return impl(path)


def _open_results(path: Path, model_h5: Path | None) -> Any:
    from apeGmsh.viewers.__main__ import _open_results as impl

    return impl(path, model_h5)
