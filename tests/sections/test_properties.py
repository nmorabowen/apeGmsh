"""Tests — ADR 0080 B6: live-properties build controller.

The controller runs document builds off the UI thread and marshals the
freshest result back. These tests need **no Qt** — the controller is
constructed with ``autostart_timer=False`` and :meth:`drain` is driven
manually, so the coalescing / memoization / staleness logic is exercised
deterministically, and the one real-``build_document`` test proves the
solve ran on a worker thread (the S6 no-solve-on-the-UI-thread law).
"""
from __future__ import annotations

import threading

import pytest

from apeGmsh.sections import SectionDocument
from apeGmsh.sections._properties import (
    BuildResult,
    PropertiesController,
    build_document,
    canonical_state,
    fiber_identities,
)


# ─────────────────────────────────────────────────────────────────────
# fiber-sum identities (exact, cheap, no solve)
# ─────────────────────────────────────────────────────────────────────


def test_fiber_identities_exact_sums():
    doc = SectionDocument.new(kind="fiber", name="f")
    doc.set_material("c", uniaxial=("Elastic", {"E": 1.0}))
    doc.set_material("s", uniaxial=("Elastic", {"E": 1.0}))
    doc.add_patch_rect(material="c", ny=2, nz=2,
                       yI=-5.0, zI=-5.0, yJ=5.0, zJ=5.0)   # 10×10 = 100
    doc.add_layer_straight(material="s", n_bars=4, area=0.5,
                           yI=-4.0, zI=-4.0, yJ=-4.0, zJ=4.0)  # 4×0.5 = 2
    ident = fiber_identities(doc.build())
    assert ident["total_area"] == pytest.approx(102.0)
    assert ident["areas_by_material"]["c"] == pytest.approx(100.0)
    assert ident["areas_by_material"]["s"] == pytest.approx(2.0)
    assert (ident["n_patches"], ident["n_layers"], ident["n_points"]) == (
        1, 1, 0,
    )


# ─────────────────────────────────────────────────────────────────────
# controller: memoization, coalescing, staleness (injected builder)
# ─────────────────────────────────────────────────────────────────────


def _doc(kind_tag: int) -> dict:
    d = SectionDocument.new(kind="continuum", name="t")
    d.set_material("m", E=float(kind_tag) + 1.0, nu=0.3)
    return d.to_dict()


def test_memoization_same_state_builds_once():
    results: list[BuildResult] = []
    calls = {"n": 0}

    def counting(doc):
        calls["n"] += 1
        return BuildResult(key="", kind="continuum")

    ctrl = PropertiesController(
        builder=counting, on_result=results.append, autostart_timer=False,
    )
    doc = _doc(1)
    ctrl.request(doc)
    ctrl.join(5.0)
    ctrl.drain()
    ctrl.request(doc)          # identical state → served from cache
    ctrl.drain()
    assert ctrl.build_count == 1
    assert calls["n"] == 1
    assert len(results) == 2   # delivered from build, then from cache


def test_coalescing_burst_collapses_to_last():
    results: list[BuildResult] = []
    release = threading.Event()
    calls = {"n": 0}

    def blocking(doc):
        calls["n"] += 1
        release.wait(5.0)
        return BuildResult(key="", kind="continuum")

    ctrl = PropertiesController(
        builder=blocking, on_result=results.append, autostart_timer=False,
    )
    a, b, c, d = (_doc(i) for i in range(4))
    ctrl.request(a)            # starts build #1 (blocks)
    ctrl.request(b)            # coalesced into pending
    ctrl.request(c)
    ctrl.request(d)            # pending is now d (last wins)
    release.set()
    ctrl.join(5.0)
    ctrl.drain()               # a dropped (stale); pending d dispatched
    ctrl.join(5.0)
    ctrl.drain()               # d delivered
    assert ctrl.build_count == 2          # ≤ 4 edits
    assert results[-1].key == canonical_state(d)
    # the stale intermediate state 'a' was never delivered
    assert all(r.key != canonical_state(a) for r in results)


def test_stale_result_not_delivered():
    results: list[BuildResult] = []
    release = threading.Event()

    def blocking(doc):
        release.wait(5.0)
        return BuildResult(key="", kind="continuum")

    ctrl = PropertiesController(
        builder=blocking, on_result=results.append, autostart_timer=False,
    )
    a, b = _doc(0), _doc(1)
    ctrl.request(a)            # build a (blocks)
    ctrl.request(b)            # latest is now b
    release.set()
    ctrl.join(5.0)
    ctrl.drain()               # a completes but is stale → not delivered
    ctrl.join(5.0)
    ctrl.drain()               # b delivered
    delivered_keys = [r.key for r in results]
    assert canonical_state(a) not in delivered_keys
    assert delivered_keys[-1] == canonical_state(b)


# ─────────────────────────────────────────────────────────────────────
# no solve on the UI thread — the real builder runs on a worker thread
# ─────────────────────────────────────────────────────────────────────


def test_build_runs_off_the_calling_thread():
    doc = SectionDocument.new(kind="continuum", name="t")
    doc.set_material("s", E=200e3, nu=0.3)
    doc.add_shape("rect_face", id="r", b=4.0, h=4.0, material="s")
    doc.set_mesh(lc=1.0)

    results: list[BuildResult] = []
    main_id = threading.get_ident()
    ctrl = PropertiesController(
        on_result=results.append, autostart_timer=False,   # real builder
    )
    ctrl.request(doc.to_dict())
    ctrl.join(60.0)
    ctrl.drain()

    assert len(results) == 1
    res = results[0]
    assert res.error is None, res.error
    assert res.worker_thread_id is not None
    assert res.worker_thread_id != main_id          # solved off this thread

    # panel-equals-headless: same analyzer numbers as a direct build()
    ref = build_document(doc.to_dict())
    assert res.analysis.geometric().area == pytest.approx(
        ref.analysis.geometric().area
    )


def test_build_document_captures_error_instead_of_raising():
    # continuum with no mesh lc → SectionDocumentError, captured as error
    doc = SectionDocument.new(kind="continuum", name="t")
    doc.add_shape("rect_face", id="r", b=2.0, h=2.0)
    res = build_document(doc.to_dict())
    assert res.error is not None
    assert res.analysis is None
