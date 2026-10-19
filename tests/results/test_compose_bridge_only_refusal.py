"""Composing from a bridge-only archive refuses with a named error (#1881).

``NativeWriter.open(fem=..., model_h5_src=...)`` copies the source's
``/opensees`` zone and forwards its ``ndm``, which it resolves from the
source's ``nodes/coords``.  A bridge-only source (``/meta`` + ``/opensees``,
no neutral zone, e.g. one written from a ``FEMStub``) has no ``nodes``, and
the compose used to leak h5py's raw ``KeyError`` after the ``/opensees``
copy had already landed.  It now raises :class:`BridgeOnlyArchiveError`
(the #1859 class) before anything is copied.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from apeGmsh.opensees.emitter.h5_reader import (
    BridgeOnlyArchiveError,
    MalformedH5Error,
)
from apeGmsh.results.writers._native import NativeWriter
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5
from tests.opensees.h5.test_bridge_only_archive_refusal import (
    _bridge_only_archive,
)


def test_composing_from_a_bridge_only_archive_raises_the_named_refusal(
    tmp_path: Path,
) -> None:
    src = _bridge_only_archive(tmp_path)
    _model, fem = build_simple_frame_h5(tmp_path)
    dst = tmp_path / "results.h5"
    writer = NativeWriter(dst)
    try:
        with pytest.raises(BridgeOnlyArchiveError) as info:
            writer.open(fem=fem, model_h5_src=src)
    finally:
        writer.close()
    msg = str(info.value)
    assert str(src) in msg
    assert "no neutral zone" in msg
    assert "Zones present: /meta, /opensees" in msg
    # In the h5 error family, not h5py's raw KeyError.
    assert isinstance(info.value, MalformedH5Error)
    assert not isinstance(info.value, KeyError)
    # Refused before the copy: no half-composed bridge zone.
    with h5py.File(dst, "r") as f:
        assert "opensees" not in f


def test_a_bridge_only_archive_without_fem_still_composes(
    tmp_path: Path,
) -> None:
    # No /model/meta to forward ndm onto, so nothing reads nodes/coords:
    # the refusal is exactly as narrow as the KeyError it replaces.
    src = _bridge_only_archive(tmp_path)
    dst = tmp_path / "results.h5"
    writer = NativeWriter(dst)
    writer.open(model_h5_src=src)
    writer.close()
    with h5py.File(src, "r") as s, h5py.File(dst, "r") as d:
        assert "model" not in d
        assert "opensees" in d
        assert sorted(d["opensees"].keys()) == sorted(s["opensees"].keys())


def test_composing_from_a_source_with_nodes_is_unchanged(
    tmp_path: Path,
) -> None:
    src, fem = build_simple_frame_h5(tmp_path)
    dst = tmp_path / "results.h5"
    writer = NativeWriter(dst)
    writer.open(fem=fem, model_h5_src=src)
    writer.close()
    with h5py.File(src, "r") as s, h5py.File(dst, "r") as d:
        assert "opensees" in d
        assert sorted(d["opensees"].keys()) == sorted(s["opensees"].keys())
        assert int(d["model/meta"].attrs["ndm"]) == int(s["meta"].attrs["ndm"])
