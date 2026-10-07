"""Compose carries the source's ``rebar_elements``, and warns iff it drops
a non-empty source stream (program slices B2-2 D9, AS2a).

``g.compose`` rebuilds the host's ``ElementComposite`` from the rewritten
bundle. Until AS2a the bundle lacked the source's ``elements.rebar_elements``
(the cage's auto-emitted structural rebar from
``g.rebar.place(emit_elements=True)``) and the rewriter warned
:class:`ComposeDroppedStreamWarning`. ADR 0117 D4 carries it: each bar cell
moves with the module's id offset, the bar PG is prefixed as every PG is,
and the material (the bond name) becomes the bridge name
``{label}.{material}``, so it binds to the instance's rehydrated material.
No stream is uncarried today; the helper still warns for any stream that
joins :data:`_UNCARRIED_ELEMENT_STREAMS`.

Run with ``-W error::apeGmsh.mesh._compose.ComposeDroppedStreamWarning`` to
prove the carry is silent. Built on a real conformal cage placement (no
fork build), like ``tests/mesh/test_rebar_element_h5_roundtrip.py``.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Cage
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.mesh._compose import (
    ComposeDroppedStreamWarning,
    _UNCARRIED_ELEMENT_STREAMS,
    _warn_dropped_streams,
)


def _rebar_module_h5(path) -> int:
    """A conformal cage with one interior bar, emit_elements=True.

    Returns the module's ``rebar_elements`` count (one per placed bar PG).
    """
    with apeGmsh(model_name="rebar_mod", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                          db=0.0254, material="rebar", name="L1")
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.to_h5(str(path))
        return len(fem.elements.rebar_elements)


def _plain_h5(path) -> None:
    with apeGmsh(model_name="plain", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(path))


def _compose_and_reload(host, mod, tmp_path, *, label: str) -> FEMData:
    g = apeGmsh.from_h5(str(host))
    g.compose(str(mod), label=label, translate=(2.0, 0.0, 0.0))
    out = tmp_path / "out.h5"
    g.save(str(out))
    return FEMData.from_h5(str(out))


# ── the source stream is carried, silently ──────────────────────────

def test_compose_carries_source_rebar_elements(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    n = _rebar_module_h5(mod)
    assert n == 1
    (bar,) = FEMData.from_h5(str(mod)).elements.rebar_elements
    _plain_h5(host)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ComposeDroppedStreamWarning)
        merged = _compose_and_reload(host, mod, tmp_path, label="A")

    (got,) = merged.elements.rebar_elements
    # The bond (bar material) name is the bridge name ``A.<material>``.
    assert got.material == f"A.{bar.material}" == "A.rebar"
    sep = "/" if "." in bar.pg else "."     # ADR 0038 PG alternation
    assert got.pg == f"A{sep}{bar.pg}"
    assert (got.element, got.area, got.role) == (bar.element, bar.area, bar.role)
    # The cells moved with the module by one offset, onto merged nodes
    # that sit above every host node.
    off = got.connectivity[0][0] - bar.connectivity[0][0]
    assert got.connectivity == tuple(
        (i + off, j + off) for i, j in bar.connectivity)
    merged_ids = {int(i) for i in merged.nodes.ids}
    host_max = max(int(i) for i in FEMData.from_h5(str(host)).nodes.ids)
    cell_nodes = {i for c in got.connectivity for i in c}
    assert cell_nodes <= merged_ids and min(cell_nodes) > host_max


# ── carried streams and empty ones stay silent ──────────────────────

def test_plain_compose_stays_silent(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    _plain_h5(mod)
    _plain_h5(host)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComposeDroppedStreamWarning)
        merged = _compose_and_reload(host, mod, tmp_path, label="B")
    assert merged.elements.rebar_elements == []


def test_host_rebar_elements_are_carried_silently(tmp_path):
    # A rebar HOST + a plain module: the host's stream is carried by the
    # merge (not dropped), so no warning fires and the count survives.
    rebar_host = tmp_path / "rebar_host.h5"
    plain_mod = tmp_path / "plain_mod.h5"
    n = _rebar_module_h5(rebar_host)
    _plain_h5(plain_mod)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComposeDroppedStreamWarning)
        merged = _compose_and_reload(rebar_host, plain_mod, tmp_path,
                                     label="C")
    assert len(merged.elements.rebar_elements) == n


# ── the helper's contract (unit, no gmsh) ───────────────────────────

class _Elems:
    def __init__(self, **streams):
        for k, v in streams.items():
            setattr(self, k, v)


class _Src:
    def __init__(self, **streams):
        self.elements = _Elems(**streams)


def test_uncarried_streams_tuple_is_empty():
    # Every ElementComposite stream is carried by _merge_bundle_into_fem
    # (rebar_elements since AS2a); a stream added here must be one the
    # bundle really lacks.
    assert _UNCARRIED_ELEMENT_STREAMS == ()


@pytest.fixture
def _one_uncarried(monkeypatch):
    """The helper's contract, on a stand-in stream named ``future``."""
    import apeGmsh.mesh._compose as compose

    monkeypatch.setattr(compose, "_UNCARRIED_ELEMENT_STREAMS", ("future",))


def test_helper_names_stream_and_count(_one_uncarried):
    src = _Src(future=[object(), object(), object()])
    with pytest.warns(ComposeDroppedStreamWarning,
                      match=r"carries 3 elements\.future"):
        _warn_dropped_streams(src, label="X")


def test_helper_silent_on_empty_stream(_one_uncarried):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComposeDroppedStreamWarning)
        _warn_dropped_streams(_Src(future=[]), label="X")


def test_helper_fails_closed_on_missing_stream(_one_uncarried):
    # The stream is a public ElementComposite attribute; an object without
    # it is a contract break, not a reason to stay quiet.
    with pytest.raises(AttributeError):
        _warn_dropped_streams(_Src(), label="X")
