"""Fork lane: FEM-id to OpenSees-tag translation on MPCO and ``.ladruno``.

Gap G18 of the DomainCapture parity table (#1767, #1945; card #1957).
DomainCapture translates FEM element ids to OpenSees tags and refuses a
record its pairing does not cover. Both HDF5 replacements key element
results by OpenSees tag and translate back through the pairing the bridge
writes to ``model.h5`` (``ElementTagTranslator``). This file proves that
translation end to end on one model where the two id spaces differ: the
solid + frame model of ``tests/test_results_recorder_tag_parity.py`` (a
gmsh solid with ``Base``/``Top`` surface groups, an elastic ``Strut`` and a
force-based ``Beam``, sequential tags), whose ``.out`` twin runs on the
live-stock lane.

The model is solved in two steps and recorded to MPCO and ``.ladruno`` in
the same run; each file is read with ``model_h5=``. For each route and each
element kind:

* **nodal forces** (``Strut``, ``globalForce``; C2 of the G9 lane audit),
* **Gauss** (``Solid``, ``stresses``),
* **line stations** (``Beam``, ``section.force``; C5 of the G9 lane audit),

the slab's element ids are the FEM ids of the group; every row equals
``eleResponse`` on the OpenSees tag paired with its label, at every step and
``rtol=1e-10``; and an ``ids=`` filter by FEM id returns exactly those rows.
The strut's base-end nodal forces also equal ``nodeReaction`` at its fixed
node (C2).

An id outside the pairing must return no rows. On both formats a Base/Top
quad id that equals an emitted OpenSees tag reads that element's rows under
the quad's id: the filter passes through untranslated (#2066). Those cases
are strict xfails.

The mutation check reads the same files with the pairing removed from the
model archive and shows the id and value assertions fail.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results
from tests.test_results_recorder_tag_parity import (
    SF_KINDS,
    OutsidePairingReturnedRows,
    _drop_pairing,
    _rotate_pairing,
    _sf_assert_filter_is_exact,
    _sf_assert_ids_and_values,
    _sf_assert_outside_returns_none,
    _sf_capture,
    _sf_groups,
    _sf_outside,
    _sf_read,
    _solid_frame_bridge,
    _solid_frame_fem,
)

pytestmark = pytest.mark.ladruno_fork

ROUTES = ("mpco", "ladruno")
N_STEPS = 2
ELEM_RESPONSES = ("globalForce", "stresses", "section.force")


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _open(route: str, path: str, fem: Any, model_h5: str) -> Results:
    reader = Results.from_mpco if route == "mpco" else Results.from_ladruno
    return reader(path, fem=fem, model_h5=model_h5)


def _ladruno_read(res: Results, kind: str, comp: str,
                  ids: Any = None) -> dict[int, Any]:
    """``_sf_read`` for ``.ladruno``. Its element level is token-driven:
    the component is the file's ``globalForce`` key and the slab holds the
    raw ``(T, E, 12)`` output vector, end i then end j. Component ``k`` of
    the nodal-forces kind is columns ``k`` and ``k + 6``."""
    if kind != "nodal_forces":
        return _sf_read(res, kind, comp, ids=ids)
    k = SF_KINDS[kind][1].index(comp)
    slab = res.elements.get(component="globalForce", ids=ids)
    vals = np.asarray(slab.values)
    assert vals.shape[1] == 0 or vals.shape[-1] == 12, vals.shape
    return {int(e): vals[:, i, [k, k + 6]]
            for i, e in enumerate(np.asarray(slab.element_ids))}


READ = {"mpco": _sf_read, "ladruno": _ladruno_read}


@pytest.fixture(scope="module")
def solved(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("g18_fork")
    fem = _solid_frame_fem()
    ops = _solid_frame_bridge(fem)
    paths = {r: str(tmp / f"solid_frame.{r}") for r in ROUTES}
    # A node channel on MPCO too: an element-only MPCO file reads 0 steps
    # (#1970).
    ops.recorder.MPCO(file=paths["mpco"],
                      nodal_responses=("displacement",),
                      elem_responses=ELEM_RESPONSES)
    ops.recorder.Ladruno(file=paths["ladruno"],
                         elem_responses=ELEM_RESPONSES)
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    fem_to_ops = _fem_to_ops(model_h5)
    groups = _sf_groups(fem)

    strut_base = int(ops.nodes.one(label="strut_base").tag)
    root = [e for e, conn in _strut_conn(fem).items() if strut_base in conn]
    assert len(root) == 1, root
    end = _strut_conn(fem)[root[0]].index(strut_base)

    live, reaction = [], []
    o = emitter.ops
    for _ in range(N_STEPS):
        assert emitter.analyze(steps=1) == 0
        live.append(_sf_capture(o, fem_to_ops, groups))
        o.reactions()
        reaction.append(np.asarray(o.nodeReaction(strut_base), dtype=float))
    o.remove("recorders")          # flush and close both files
    # Every kind carries non-zero values, so a zero read cannot pass.
    for kind, per_e in live[-1].items():
        assert all(np.abs(v).max() > 0.0 for v in per_e.values()), kind
    return dict(
        fem=fem, model_h5=model_h5, paths=paths, groups=groups,
        fem_to_ops=fem_to_ops, live=live,
        outside=_sf_outside(fem, fem_to_ops),
        root=(root[0], end), reaction=np.stack(reaction),
        results={r: _open(r, paths[r], fem, model_h5) for r in ROUTES},
    )


def _strut_conn(fem: Any) -> dict[int, list[int]]:
    strut = set(_sf_groups(fem)["nodal_forces"])
    return {int(e): [int(n) for n in conn]
            for grp in fem.elements
            for e, conn in zip(grp.ids, grp.connectivity)
            if int(e) in strut}


def _params() -> list[Any]:
    return [pytest.param(r, k, id=f"{r}-{k}") for r in ROUTES for k in SF_KINDS]


# ---------------------------------------------------------------------------
# Ids and values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route, kind", _params())
def test_rows_are_the_paired_live_responses(
    solved: dict[str, Any], route: str, kind: str,
) -> None:
    s = solved
    for e in s["groups"][kind]:
        assert s["fem_to_ops"][e] != e                    # not vacuous
    res = s["results"][route]
    assert res.stages[0].n_steps == N_STEPS
    _sf_assert_ids_and_values(res, kind, s["groups"], s["live"],
                              s["fem_to_ops"], route, read=READ[route])


@pytest.mark.parametrize("route", ROUTES)
def test_strut_base_nodal_forces_equal_the_reaction(
    solved: dict[str, Any], route: str,
) -> None:
    """C2: the root strut's nodal resisting forces at its fixed end are the
    node's reaction, read back by FEM id."""
    eid, end = solved["root"]
    res = solved["results"][route]
    comps = SF_KINDS["nodal_forces"][1]
    got = np.stack([READ[route](res, "nodal_forces", c, ids=[eid])[eid][:, end]
                    for c in comps], axis=1)               # (T, 6)
    want = solved["reaction"]
    np.testing.assert_allclose(got, want, rtol=1e-10,
                               atol=1e-10 * float(np.abs(want).max()))


@pytest.mark.parametrize("route, kind", _params())
def test_fem_id_filter_is_exact(
    solved: dict[str, Any], route: str, kind: str,
) -> None:
    _sf_assert_filter_is_exact(solved["results"][route], kind,
                               solved["groups"], read=READ[route])


@pytest.mark.xfail(strict=True, raises=OutsidePairingReturnedRows,
                   reason="#2066: an ids= filter with an id outside the "
                          "pairing passes through untranslated, so a FEM id "
                          "equal to an emitted ops tag reads that element")
@pytest.mark.parametrize("route, kind", _params())
def test_id_outside_the_pairing_returns_none(
    solved: dict[str, Any], route: str, kind: str,
) -> None:
    _sf_assert_outside_returns_none(solved["results"][route], kind,
                                    solved["outside"], read=READ[route])


# ---------------------------------------------------------------------------
# Mutation checks: the same files read through a broken pairing
# ---------------------------------------------------------------------------


def _mutated_read(s: dict[str, Any], tmp_path: Path, route: str,
                  mutate: Any) -> Results:
    path = str(tmp_path / "model_mutated.h5")
    shutil.copyfile(s["model_h5"], path)
    with h5py.File(path, "r+") as f:
        mutate(f["/opensees"])
    return _open(route, s["paths"][route], s["fem"], path)


@pytest.mark.parametrize("route, kind", _params())
def test_oracle_rejects_a_read_without_the_pairing(
    solved: dict[str, Any], tmp_path: Path, route: str, kind: str,
) -> None:
    """With ``element_meta`` deleted from a copy of the model archive, ids
    pass through untranslated: the read warns that it has no pairing, and
    the id assertion must fail."""
    res = _mutated_read(solved, tmp_path, route, _drop_pairing)
    s = solved
    with pytest.warns(UserWarning, match="no fem_eid<->ops-tag pairing"), \
            pytest.raises(AssertionError, match=f"{route} {kind} .* ids"):
        _sf_assert_ids_and_values(res, kind, s["groups"], s["live"],
                                  s["fem_to_ops"], route, read=READ[route])


@pytest.mark.parametrize("route, kind", _params())
def test_oracle_rejects_a_permuted_pairing(
    solved: dict[str, Any], tmp_path: Path, route: str, kind: str,
) -> None:
    """With each family's ``fem_eids`` rotated by one in a copy of the
    model archive, every id is still a group member, so the id assertion
    passes and the value assertion must fail."""
    res = _mutated_read(solved, tmp_path, route, _rotate_pairing)
    s = solved
    with pytest.raises(AssertionError, match="of FEM element"):
        _sf_assert_ids_and_values(res, kind, s["groups"], s["live"],
                                  s["fem_to_ops"], route, read=READ[route])
