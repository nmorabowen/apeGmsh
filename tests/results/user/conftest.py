"""Shared builders for the ``.user.h5`` tests (ADR 0130 VU1)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import pytest

from apeGmsh.results.user import StateAddress, UserResult

REPO = Path(__file__).resolve().parents[3]
LADRUNO = REPO / "tests" / "fixtures" / "ladruno"
FEM_HASH = "fedcba9876543210fedcba9876543210"
MODEL_HASH = "0123456789abcdef0123456789abcdef"


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


def nodal(name: str = "Delta_after (L/480)", *, basis: str = "model", **kw) -> UserResult:
    """A two-state nodal field with a NaN cell: ``D (2, 3, 2)``."""
    data = np.array(
        [[[1.5, -2.5], [np.nan, 1e-300], [3.0, 4.0]],
         [[0.1, 0.2], [0.3, 0.4], [np.nan, np.nan]]], dtype="f8",
    )
    fields = dict(
        name=name, target="nodes", basis=basis,
        states=(StateAddress("stage", "gravity", -1, 0), StateAddress("combination", "1.2D+1.6L")),
        components=("wx", "wy"), units=("mm", "mm"), data=data,
        ids=np.array([1, 2, 3], dtype="i4"),
    )
    fields.update(kw)
    return UserResult(**fields)
