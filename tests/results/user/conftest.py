"""Fixtures for the ``.user.h5`` tests (ADR 0130 VU1); builders are in ``_user_helpers``."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import h5py
import pytest

from tests.results.user._user_helpers import FEM_HASH, LADRUNO, MODEL_HASH


@pytest.fixture
def make_model(tmp_path: Path) -> Callable[..., Path]:
    """A minimal archive holding only what the pairing reads: ``/meta/lineage``."""

    def make(
        name: str = "model.h5", fem: str = FEM_HASH, model: Optional[str] = MODEL_HASH,
    ) -> Path:
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            lineage = f.create_group("meta/lineage")
            lineage.attrs["fem_hash"] = fem
            if model is not None:
                lineage.attrs["model_hash"] = model
        return path

    return make


@pytest.fixture
def ladruno(tmp_path: Path) -> Path:
    """A private copy of the truss fixture, free to be stamped or rewritten."""
    path = tmp_path / "run.ladruno"
    shutil.copy(LADRUNO / "truss2d.ladruno", path)
    return path
