"""shell_probe: warn when a recorder reads the upstream shells' stale state.

The shells in :data:`~apeGmsh.opensees._response_catalog.ZERO_GAUSS_PROBE_CLASSES`
set their sections' trial strain only when the residual is formed, and
``algorithm Linear`` never forms it after its solve. Every probe of the
section state after a Linear step then answers stale values: zero at step
1, then the previous step's state (ShellMITC4 / ShellDKGQ / ShellDKGT) or
zero again (ShellNLDKGQ / ShellNLDKGT). ``eleResponse``, a ``.out``
Element recorder and MPCO agree (measured in
``tests/test_results_shell_resultant_vocabulary.py``), so a recorder
writes them without any error. This module makes the emit say so.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from ..errors import BridgeError

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData

    from ...types import Recorder
    from ..recorders import ElementPlan


class StaleShellProbeWarning(UserWarning):
    """A recorder reads an upstream shell's section state after a step
    solved with ``algorithm Linear``, where that state is stale."""


#: Element responses that answer nodal forces: ShellMITC4 / ShellDKGQ /
#: ShellDKGT form them from the current displacement (correct under
#: Linear, measured on stock and fork); ShellNLDKGQ / ShellNLDKGT answer
#: zero under Linear, so for those classes they are stale too.
_NODAL_FORCE_TOKENS: frozenset[str] = frozenset({
    "force", "forces", "globalForce", "globalForces",
    "localForce", "localForces",
})

#: The listed classes whose nodal forces are stale under Linear as well.
_FORCES_ALSO_STALE: frozenset[str] = frozenset({"ShellNLDKGQ", "ShellNLDKGT"})

_REMEDY = (
    "Solve with algorithm Newton instead (ops.algorithm.Newton(), with a "
    "convergence test): on a linear problem it converges in one iteration "
    "and leaves the sections at the converged state, so every route reads "
    "the right values. Or build the shells as ASDShellQ4 / ASDShellT3."
)


def _reads_stale_state(class_name: str, token: str) -> bool:
    """Whether recording ``token`` on ``class_name`` reads stale state
    after a Linear step. Any token other than a nodal force reads the
    sections (fail closed: an unmeasured token counts as stale)."""
    if token in _NODAL_FORCE_TOKENS:
        return class_name in _FORCES_ALSO_STALE
    return True


def _is_linear(algorithm: object) -> bool:
    from ....analysis.algorithm import Linear

    return isinstance(algorithm, Linear)


def _listed_targets(
    element_plan: "ElementPlan",
) -> "dict[str, tuple[np.ndarray, np.ndarray]]":
    """``{class: (fem_eids, ops_tags)}`` for every listed class the plan
    emits."""
    from ...._response_catalog import ZERO_GAUSS_PROBE_CLASSES

    acc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for spec, rows in element_plan:
        cls = type(spec).__name__
        if cls not in ZERO_GAUSS_PROBE_CLASSES or not len(rows):
            continue
        tags = (
            rows.tags if rows.tags is not None
            else rows.tag_start + np.arange(len(rows), dtype=np.int64)
        )
        fem, ops = acc.setdefault(cls, ([], []))
        fem.append(np.asarray(rows.eids, dtype=np.int64))
        ops.append(np.asarray(tags, dtype=np.int64))
    return {
        cls: (np.concatenate(fem), np.concatenate(ops))
        for cls, (fem, ops) in acc.items()
    }


def _element_records(
    rec: "Recorder", fem: "FEMData",
) -> "list[tuple[str, str | None, Sequence[int] | None, tuple[str, ...]]]":
    """The element-level reads of one recorder.

    Each entry is ``(description, id_space, ids, tokens)``: ``id_space``
    is ``"fem"`` (FEM element ids) or ``"ops"`` (OpenSees tags), and
    ``ids`` is ``None`` for a whole-model recorder. A recorder that reads
    no element response yields nothing; a recorder class this function
    does not know raises, so a new one is classified here.
    """
    from ....recorder import (
        Element as ElementRecorder,
        FilterableRecorder,
        Monitor,
        Node,
        RecorderDeclaration,
    )
    from ...._recorder_translate import element_record_response_tokens
    from ..pg_expand import expand_pg_to_elements
    from ..recorders import _resolve_element_targets

    if isinstance(rec, (Node, Monitor)):
        return []
    if isinstance(rec, ElementRecorder):
        desc = f"recorder Element (file {rec.file!r})"
        if rec.pg is not None:
            ids = tuple(eid for eid, _c in expand_pg_to_elements(fem, rec.pg))
            return [(desc, "fem", ids, rec.response)]
        space = "fem" if rec.label is not None else "ops"
        return [(desc, space, rec.elements, rec.response)]
    if isinstance(rec, FilterableRecorder):
        if not rec.elem_responses:
            return []
        desc = f"recorder {type(rec).__name__} (file {rec.file!r})"
        if rec.elements_pg is not None:
            ids = tuple(
                eid for eid, _c in expand_pg_to_elements(fem, rec.elements_pg))
            return [(desc, "fem", ids, rec.elem_responses)]
        if rec.elements is not None:
            return [(desc, "ops", rec.elements, rec.elem_responses)]
        return [(desc, None, None, rec.elem_responses)]
    if isinstance(rec, RecorderDeclaration):
        out: list[tuple[str, str | None, Sequence[int] | None, tuple[str, ...]]] = []
        for record in rec.records:
            if record.category not in ("elements", "gauss", "line_stations"):
                continue
            # The response phrase the emit writes for this record (the
            # same translator; ``sigma_zz_capable=None`` keeps it silent).
            canonical = element_record_response_tokens(
                record.category, record.components, record_name=record.name,
                sigma_zz_capable=None,
            ) if record.components else None
            tokens = (*(canonical or ()), *record.raw)
            if not tokens:
                continue
            out.append((
                f"recorder.declare({rec.name!r}) record "
                f"{record.name or 'default'!r} ({record.category})",
                "fem", _resolve_element_targets(record, fem), tokens,
            ))
        return out
    raise BridgeError(
        f"warn_stale_shell_probe_under_linear: recorder class "
        f"{type(rec).__name__} is not classified; say which element "
        "responses it reads before it can be emitted beside a Linear "
        "analysis of a ZERO_GAUSS_PROBE_CLASSES shell."
    )


def warn_stale_shell_probe_under_linear(
    fem: "FEMData",
    element_plan: "ElementPlan",
    recorders: "Iterable[Recorder]",
    *,
    staged: bool,
    flat_algorithms: "Sequence[object]",
    stage_chain: "Sequence[tuple[str, Sequence[object], frozenset[int]]]",
) -> None:
    """Warn once per (recorder record, class) that reads stale shell state.

    ``flat_algorithms`` and each stage's algorithms list every algorithm
    the domain can step with: the chain's own and every rung of its
    strategy ladder (a rescue rung solves the step too).
    ``stage_chain`` is each stage in order as ``(name, algorithms,
    claimed recorder ids)``. In a flat deck a recorder sees the flat
    algorithms; in a staged deck a recorder claimed by stage *k* sees
    stages *k* onward and an unclaimed one sees every stage. A recorder
    that sees a Linear step and reads a non-force response (or, for the
    NL classes, any response) of a listed class warns.

    The other Linear-algorithm gate runs first, on the same facts: a
    multi-yield soil material under a Linear step raises
    (:func:`~.soil.refuse_multi_yield_soil_under_linear`).
    """
    from .soil import refuse_multi_yield_soil_under_linear

    refuse_multi_yield_soil_under_linear(
        (spec for spec, _rows in element_plan), staged=staged,
        flat_algorithms=flat_algorithms, stage_chain=stage_chain)
    if staged:
        linear_stages = [
            i for i, (_n, algos, _ids) in enumerate(stage_chain)
            if any(_is_linear(a) for a in algos)
        ]
        if not linear_stages:
            return
    elif not any(_is_linear(a) for a in flat_algorithms):
        return
    listed = _listed_targets(element_plan)
    if not listed:
        return

    claimed_at = {
        rid: k for k, (_n, _a, ids) in enumerate(stage_chain) for rid in ids
    }
    for rec in recorders:
        if staged:
            first = claimed_at.get(id(rec), 0)
            seen = [stage_chain[i][0] for i in linear_stages if i >= first]
            if not seen:
                continue
            where = "stage(s) " + ", ".join(repr(s) for s in seen)
        else:
            where = "the analysis"
        for desc, space, ids, tokens in _element_records(rec, fem):
            for cls, (fem_eids, ops_tags) in sorted(listed.items()):
                if not any(_reads_stale_state(cls, t) for t in tokens):
                    continue
                if ids is None:
                    hit = len(fem_eids)
                else:
                    pool = fem_eids if space == "fem" else ops_tags
                    hit = int(np.isin(
                        pool, np.asarray(ids, dtype=np.int64)).sum())
                if not hit:
                    continue
                warnings.warn(
                    f"{desc} records {' '.join(tokens)!r} on {hit} {cls} "
                    f"element(s), and {where} solves with algorithm Linear. "
                    f"Linear never re-forms the residual after its solve, so "
                    f"{cls} sections never see the converged displacement: "
                    "the recorded values are stale, zero at step 1 and then "
                    "the previous step's state (or zero again for the NL "
                    "shells), with no error from OpenSees. " + _REMEDY,
                    StaleShellProbeWarning,
                    stacklevel=4,
                )
