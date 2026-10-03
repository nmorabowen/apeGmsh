"""STKO ``.scd`` -> apeGmsh session -> apeSees bridge, end to end (ADR 0111).

The integrator of the three translator modules
(``internal_docs/stko_translator_rules.md`` section 9):

* :func:`collect_unsupported` -- the union of the three registry slices
  (mesh, properties, conditions), merged and sorted. Empty means the
  document translates.
* :func:`translate_scd` -- the session half: STKO's own mesh (same node and
  element ids) as discrete entities and physical groups, the conditions plan,
  and the rigid diaphragms on ``g.constraints``. Raises
  :class:`~.translate_types.UnsupportedSTKOTypes`, listing every offender,
  before the session is touched.
* :func:`build_opensees` -- the bridge half: ``apeSees(fem)``, then the
  properties and elements (:func:`~.translate_props.build_props`) and the
  conditions (:func:`~.translate_conditions.build_conditions`).

Nothing is exported here; the caller picks ``ops.tcl`` / ``ops.py`` /
``ops.h5``. ``build_props`` and ``build_conditions`` stay public so a recipe
can use one without the other.

Element tags: STKO's tag is the mesh element id. :func:`build_opensees`
builds the bridge with ``apeSees(fem, element_tags="fem")`` (ADR 0111 D2,
rules section 9.1), so the deck's element tags are STKO's ids; tags the
bridge synthesises (interface springs, couplings) land above the largest
FEM id. ``element_tags="sequential"`` keeps the bridge's own numbering.
Node tags are STKO's in either case.
"""
from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from . import translate_conditions, translate_mesh, translate_props
from .model import ScdModel
from .read_scd import read_scd
from .translate_types import (
    ConditionSummary,
    PropsResult,
    TranslateResult,
    Unsupported,
    UnsupportedSTKOTypes,
)

__all__ = [
    "build_opensees",
    "collect_unsupported",
    "declare_translation",
    "translate_scd",
]


def collect_unsupported(scd: ScdModel) -> tuple[Unsupported, ...]:
    """Every STKO type or option this version will not translate.

    The union of :func:`translate_mesh.check_supported`,
    :func:`translate_props.check_supported` and
    :func:`translate_conditions.check_supported`. Entries with the same
    ``(category, xobj_meta, reason)`` are merged (ids and names unioned);
    the result is sorted by that key.
    """
    merged: dict[tuple[str, str, str], tuple[set[int], set[str]]] = {}
    for check in (translate_mesh.check_supported, translate_props.check_supported,
                  translate_conditions.check_supported):
        for u in check(scd):
            ids, names = merged.setdefault((u.category, u.xobj_meta, u.reason), (set(), set()))
            ids.update(u.ids)
            names.update(u.names)
    return tuple(
        Unsupported(category=cat, xobj_meta=meta, ids=tuple(sorted(ids)),  # type: ignore[arg-type]
                    names=tuple(sorted(names)), reason=reason)
        for (cat, meta, reason), (ids, names) in sorted(merged.items())
    )


def translate_scd(
    g: Any,
    scd: ScdModel | str | Path,
    *,
    records: Mapping[int, Sequence[float]] | None = None,
) -> TranslateResult:
    """Put an STKO document into the empty session ``g``.

    1. :func:`collect_unsupported`; raise :class:`UnsupportedSTKOTypes` if
       anything is listed (the session is untouched).
    2. :func:`translate_conditions.plan_conditions` -- masses, loads, fixities,
       diaphragm pairs, time series, patterns, Rayleigh, stages (pure data;
       it runs before the session is touched, so a bad ``records=`` raises
       on an empty session).
    3. :func:`translate_mesh.build_mesh` -- STKO's nodes and elements with
       their own ids, physical groups per element group / selection set /
       condition, diaphragm carriers (the same groups as the plan's pairs:
       :func:`translate_conditions.diaphragm_groups`).
    4. :func:`translate_conditions.declare_session` -- the rigid diaphragms on
       ``g.constraints``.

    ``records`` maps a ``timeSeries.Path`` definition id to its values (the
    San Ramon ``.scd`` files hold one-value placeholders).

    Afterwards take the FEM with ``g.mesh.queries.get_fem_data(dim=None)``;
    never call ``generate()`` or ``renumber()`` on this session.
    """
    if not isinstance(scd, ScdModel):
        scd = read_scd(scd)
    problems = collect_unsupported(scd)
    if problems:
        raise UnsupportedSTKOTypes(problems)
    plan = translate_conditions.plan_conditions(scd, records=records)
    mesh = translate_mesh.build_mesh(g, scd)
    translate_conditions.declare_session(g, scd, plan, mesh)
    summaries = tuple(
        _with_pgs(s, mesh.condition_pgs.get(s.stko_id, {})) for s in plan.summaries
    )
    plan = dataclasses.replace(plan, summaries=summaries)
    return TranslateResult(scd=scd, mesh=mesh, plan=plan, ignored=plan.ignored)


def _with_pgs(s: ConditionSummary, pgs: Mapping[str, str]) -> ConditionSummary:
    return dataclasses.replace(s, pgs=dict(pgs))


def _bridge_takes_element_tags() -> bool:
    from ...opensees.apesees import apeSees
    return "element_tags" in inspect.signature(apeSees.__init__).parameters


def build_opensees(
    fem: Any,
    result: TranslateResult,
    *,
    element_tags: Literal["fem", "sequential"] = "fem",
    stages: bool = True,
    chain: Literal["serial", "stko"] = "serial",
) -> Any:
    """Declare a translated document on a new ``apeSees`` bridge.

    ``fem`` is ``g.mesh.queries.get_fem_data(dim=None)`` of the session
    :func:`translate_scd` filled. Steps: ``apeSees(fem)`` and
    ``ops.model(ndm=3, ndf=6)``; :func:`translate_props.build_props` (one
    element declaration per element group, every reachable section and
    material once); :func:`translate_conditions.build_conditions` (fixities,
    nodal masses, Rayleigh, time series, and with ``stages=True`` one
    ``ops.stage`` per static STKO stage; ``stages=False`` declares the Plain
    patterns globally and no analysis).

    ``element_tags="fem"`` (the default) gives every element its STKO mesh
    id as OpenSees tag (``apeSees(fem, element_tags="fem")``);
    ``"sequential"`` keeps the bridge's own numbering. A bridge without that
    option raises ``TypeError`` rather than silently renumbering. ``chain`` is
    :func:`translate_conditions.build_conditions`'s solver-chain mapping
    (``"serial"`` swaps STKO's MPI-only choices for serial ones).

    The transient stage, UniformExcitation patterns and STKO's adaptive
    time-step driver are not declared: they are data in ``result.plan``
    (the time-history driver owns them). A caller that needs the created
    time series or primitives builds the bridge itself and calls
    :func:`declare_translation`.

    Returns the bridge; nothing is written.
    """
    from ...opensees.apesees import apeSees

    if element_tags not in ("fem", "sequential"):
        raise ValueError(f"element_tags must be 'fem' or 'sequential', not {element_tags!r}")
    if not _bridge_takes_element_tags():
        raise TypeError(
            "this apeSees has no element_tags= option (ADR 0111 D2): the translated deck "
            "cannot carry STKO's element ids; update apeGmsh")
    ops = apeSees(fem, element_tags=element_tags)
    ops.model(ndm=3, ndf=6)
    declare_translation(ops, result, stages=stages, chain=chain)
    return ops


def declare_translation(
    ops: Any,
    result: TranslateResult,
    *,
    stages: bool = True,
    chain: Literal["serial", "stko"] = "serial",
) -> tuple[PropsResult, dict[int, Any]]:
    """Steps 6-7 of :func:`build_opensees` on a bridge the caller made
    (``ops.model(ndm=3, ndf=6)`` already declared).

    Returns ``(props, series)``: the :class:`~.translate_types.PropsResult`
    and the created time series by STKO definition id (a time-history driver
    builds its UniformExcitation patterns on them).
    """
    props = translate_props.build_props(ops, result.scd, result.mesh.element_groups)
    series = translate_conditions.build_conditions(
        ops, result.plan, result.mesh, stages=stages, chain=chain)
    return props, series
