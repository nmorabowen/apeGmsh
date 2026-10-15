"""Fork lane: RESPONSE_CATALOG against a real solve, read back from MPCO and ``.ladruno``.

Gap G6 of the DomainCapture parity table (#1767, #1945, slice #1953). The
stock-lane half reads the same one-element models from ``.out``:
``tests/test_results_catalog_out_real.py``. That file owns the case table,
the model builders and both oracles: the live ``eleResponse`` vector, and the
``D B(xi) u`` or section anchor. This file only changes the route:

* ``MPCO-spec``: the spec's own emit, ``ResolvedRecorderSpec.emit_mpco``,
  with a nodal record too.
* ``MPCO`` and ``Ladruno``: ``recorder mpco|ladruno f -N displacement -E
  <token>``, the grammar the bridge's ``ops.recorder.mpco`` and
  ``ops.recorder.ladruno`` write.

Every file is read through the public factory at ``rtol=1e-10``. The values
and the GP natural coordinates are separate tests, so a route that gets one
right and the other wrong stays pinned on both. Every class x route cell
either passes or is a strict xfail that cites a chain-B issue and asserts
the right answer:

================= ============ ============ ===============================
Family            MPCO-spec    MPCO         Ladruno
================= ============ ============ ===============================
FourNodeQuad,     pass         pass         pass
Tri31, SSPquad,
solids
SixNodeTri,       pass         pass         values pass; coords xfail #1998
EightNodeQuad     (rule 0)     (rule 0)     (rule 0: no natural coords)
shells (8 cases)  xfail #1991  pass         xfail #1990
trusses           G7           G7           G7
================= ============ ============ ===============================

The fork writes SixNodeTri and EightNodeQuad under the bracket key
``[0:0:0]``. MPCO resolves rule 0 to the class's one catalogued rule
(#1970). On ``.ladruno`` the values read, but the named-column bucket gets
no natural coordinates: #1969's catalog fallback covers only generic
``C1..Cn`` buckets (#1998). Trusses are not repeated:
``test_element_families_ladruno_live.py`` (G7, #1949) already pins them on
both routes.

Gated by the ``ladruno_fork`` marker, so the nightly ``live-fork`` lane runs it
against the pinned build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_results_catalog_out_real import (
    CASES,
    SHELL,
    Case,
    MinimalFem,
    assert_catalog_entry,
    assert_coords_match,
    assert_values_match,
    capture_truth,
    get_live_ops,
    record_components,
    solve,
)

pytestmark = pytest.mark.ladruno_fork

ROUTES = ("MPCO-spec", "MPCO", "Ladruno")
#: Written under integration rule 0 by the fork.
RULE_ZERO = frozenset({"SixNodeTri", "EightNodeQuad"})

B_SHELL_SPEC = ("#1991: emit_mpco records shell resultants as section.force; "
                "the read is empty")
B_SHELL_LADRUNO = "#1990: .ladruno shell resultants (p11..q2) read an empty slab"
B_RULE0_LADRUNO = ("#1998: .ladruno rule-0 named-column Gauss slab has no "
                   "natural coords")

#: Trusses are G7's (see the module docstring).
FORK_CASES = [k for k, c in CASES.items() if c.anchor != "truss"]


def _values_xfail(case: Case, route: str) -> "str | None":
    if case.comps == SHELL:
        return {"MPCO-spec": B_SHELL_SPEC, "Ladruno": B_SHELL_LADRUNO}.get(route)
    return None


def _coords_xfail(case: Case, route: str) -> "str | None":
    if route == "Ladruno" and case.cpp_class in RULE_ZERO:
        return B_RULE0_LADRUNO
    return _values_xfail(case, route)


def _params(xfail_for: Any) -> list[Any]:
    out = []
    for key in FORK_CASES:
        for route in ROUTES:
            reason = xfail_for(CASES[key], route)
            marks = [] if reason is None else [pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=reason)]
            out.append(pytest.param(key, route, id=f"{key}-{route}", marks=marks))
    return out


def _run(case: Case, route: str, tmp_path: Path, stub_model_h5: Path):
    from apeGmsh.results import Results
    from apeGmsh.results._bind import ModelFemMismatchWarning
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    ops = get_live_ops()
    built = case.build(ops)
    assert ops.eleType(built.eid) == case.cpp_class
    solve(ops)
    if route == "MPCO-spec":
        fem = MinimalFem(built)
        path = tmp_path / "run.mpco"
        spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
            ResolvedRecorderRecord(
                category="gauss", name="rec",
                components=record_components(case), dt=None, n_steps=None,
                element_ids=np.array([built.eid]),
                element_class_name=case.cpp_class,
            ),
            ResolvedRecorderRecord(
                category="nodes", name="u", components=("displacement_x",),
                dt=None, n_steps=None, node_ids=built.node_ids,
            ),
        ))
        with spec.emit_mpco(str(path), ops=ops):
            assert ops.analyze(1) == 0
            truth = capture_truth(ops, case, built)
    else:
        path = tmp_path / ("run.mpco" if route == "MPCO" else "run.ladruno")
        ops.recorder(route.lower(), str(path), "-N", "displacement",
                     "-E", case.token)
        assert ops.analyze(1) == 0
        truth = capture_truth(ops, case, built)
        ops.remove("recorders")
    # The stub model.h5 is another model's archive: the MPCO factory says
    # so and binds the FEMData it synthesizes from the file instead.
    if route == "Ladruno":
        r = Results.from_ladruno(str(path), model_h5=stub_model_h5)
    else:
        with pytest.warns(ModelFemMismatchWarning):
            r = Results.from_mpco(str(path), model_h5=stub_model_h5)
    assert [s.n_steps for s in r.stages] == [1]
    return built, truth, r


@pytest.mark.parametrize("key,route", _params(_values_xfail))
def test_catalog_values_read_back_from_hdf5(
    key: str, route: str, tmp_path: Path, stub_model_h5: Path,
) -> None:
    case = CASES[key]
    assert_catalog_entry(case)
    built, truth, r = _run(case, route, tmp_path, stub_model_h5)
    assert_values_match(case, built, truth,
                        lambda c: r.elements.gauss.get(component=c), where=route)


@pytest.mark.parametrize("key,route", _params(_coords_xfail))
def test_catalog_gp_coords_read_back_from_hdf5(
    key: str, route: str, tmp_path: Path, stub_model_h5: Path,
) -> None:
    case = CASES[key]
    built, truth, r = _run(case, route, tmp_path, stub_model_h5)
    assert_coords_match(case, built, truth,
                        lambda c: r.elements.gauss.get(component=c), where=route)
