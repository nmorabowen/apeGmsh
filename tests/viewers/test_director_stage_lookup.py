"""``ResultsDirector.set_stage(id)`` lands on that id, whatever the stages are named (#1393).

#1393 brings the program's ``ops.stage(name=...)`` names into the stage
namespace an ``.mpco`` exposes.  Every ``set_stage`` caller hands over
``StageInfo.id`` (``render.py``, ``web_viewer.py``, ``_session_apply.py``,
``ui/_outline_tree.py``), so the director must resolve the exact id
before a name: a program named ``stage_1`` / ``stage_2`` would otherwise
pull ``set_stage("stage_1")`` onto ``stage_0`` by name and scope every
diagram to the wrong stage.  Headless: no scene, no backend, no window.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.viewers.diagrams._director import ResultsDirector

from tests.test_results_mpco_model_h5_binding import (
    NODE_IDS,
    _ux,
    _write_model_h5,
    _write_mpco,
)


@pytest.fixture
def id_like_results(tmp_path: Path):
    """Program names that collide with the reader's ids.

    Opening it warns ``ShadowedStageNameWarning`` by design (pinned in
    ``tests/test_results_mpco_model_h5_binding.py``); these tests are
    about where ``set_stage`` lands, so the warning is silenced here.
    """
    from apeGmsh.results._bind import ShadowedStageNameWarning

    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=2)
    model_h5, _fem = _write_model_h5(
        tmp_path / "idlike.h5", stage_names=("stage_1", "stage_2"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShadowedStageNameWarning)
        r = Results.from_mpco(mpco, model_h5=model_h5)
    with r:
        assert [(s.id, s.name) for s in r.stages] == [
            ("stage_0", "stage_1"), ("stage_1", "stage_2"),
        ]
        yield r


def test_set_stage_by_last_stage_id_scopes_that_stage(id_like_results) -> None:
    """``render.py`` opens on ``stages[-1].id``: it must be the last stage."""
    r = id_like_results
    director = ResultsDirector(r)
    last = r.stages[-1]
    director.set_stage(last.id)
    assert director.stage_id == last.id == "stage_1"
    scoped = director._scoped_results()
    assert scoped.name == "stage_2"
    assert scoped.n_steps == last.n_steps == 2
    # The director lands on the last step of the stage it scoped, and
    # that stage's data is MODEL_STAGE[2]'s (closed-form values).
    assert director.step_index == last.n_steps - 1
    slab = scoped.nodes.get(component="displacement_x", ids=NODE_IDS)
    np.testing.assert_allclose(
        slab.values[-1], [_ux(2, 1, int(n)) for n in slab.node_ids],
    )


def test_set_stage_resolves_name_and_alias_after_ids(id_like_results) -> None:
    r = id_like_results
    director = ResultsDirector(r)
    director.set_stage("stage_0")
    assert director.stage_id == "stage_0"
    director.set_stage("stage_2")          # a name no id claims
    assert director.stage_id == "stage_1"
    director.set_stage("MODEL_STAGE[1]")   # the file's alias
    assert director.stage_id == "stage_0"
    with pytest.raises(KeyError, match="No stage matches"):
        director.set_stage("MODEL_STAGE[3]")
