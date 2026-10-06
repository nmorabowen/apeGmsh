"""Compose warns iff it drops a non-empty source stream (program slice
B2-2, D9).

``g.compose`` rebuilds the host's ``ElementComposite`` from the rewritten
bundle. The bundle carries every neutral-zone stream but the source's
``elements.rebar_elements`` (the cage's auto-emitted structural rebar from
``g.rebar.place(emit_elements=True)``), which used to vanish without a
word. Now the rewriter emits one :class:`ComposeDroppedStreamWarning` per
non-empty uncarried stream, naming it and its count; a plain compose and a
host-side rebar stream (which IS carried) stay silent.

Run with ``-W error::apeGmsh.mesh._compose.ComposeDroppedStreamWarning`` to
prove the silent cases are silent. Built on a real conformal cage placement
(no fork build), like ``tests/mesh/test_rebar_element_h5_roundtrip.py``.
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


# ── the drop is loud ────────────────────────────────────────────────

def test_compose_warns_when_source_rebar_elements_are_dropped(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    n = _rebar_module_h5(mod)
    assert n == 1
    assert len(FEMData.from_h5(str(mod)).elements.rebar_elements) == n
    _plain_h5(host)

    with pytest.warns(ComposeDroppedStreamWarning) as rec:
        merged = _compose_and_reload(host, mod, tmp_path, label="A")

    msgs = [str(w.message) for w in rec
            if w.category is ComposeDroppedStreamWarning]
    assert len(msgs) == 1                       # one line per dropped stream
    assert "compose(label='A')" in msgs[0]
    assert f"carries {n} elements.rebar_elements" in msgs[0]
    # The warning is honest: the stream really is absent on the result.
    # The day the bundle carries it, this assertion flips AND
    # "rebar_elements" leaves _UNCARRIED_ELEMENT_STREAMS together.
    assert len(merged.elements.rebar_elements) == 0


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


def test_uncarried_streams_tuple_names_rebar_elements_only():
    # Every other ElementComposite stream is carried by _merge_bundle_into_fem;
    # a stream added here must be one the bundle really lacks.
    assert _UNCARRIED_ELEMENT_STREAMS == ("rebar_elements",)


def test_helper_names_stream_and_count():
    src = _Src(rebar_elements=[object(), object(), object()])
    with pytest.warns(ComposeDroppedStreamWarning,
                      match=r"carries 3 elements\.rebar_elements"):
        _warn_dropped_streams(src, label="X")


def test_helper_silent_on_empty_stream():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComposeDroppedStreamWarning)
        _warn_dropped_streams(_Src(rebar_elements=[]), label="X")


def test_helper_fails_closed_on_missing_stream():
    # The stream is a public ElementComposite attribute; an object without
    # it is a contract break, not a reason to stay quiet.
    with pytest.raises(AttributeError):
        _warn_dropped_streams(_Src(), label="X")
