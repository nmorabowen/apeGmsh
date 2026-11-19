"""``results.plot`` consumption of the Ladruno-only recorder channels.

``results.energy()`` (recorder ``-G energy``) and
``results.node_envelope()`` (recorder ``-envelope``) had reader surfaces
but no rendering path anywhere — the recorder output was never consumed.
These lock the new ``plot.energy`` / ``plot.node_envelope`` methods.
GPU-free (matplotlib Agg); they assert artists/data, not pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from apeGmsh.results import Results

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
ENERGY = FIXTURES / "energy.ladruno"
NODE_ENVELOPE = FIXTURES / "node_envelope.ladruno"
TRUSS = FIXTURES / "truss2d.ladruno"
MPCO_FRAME = (
    Path(__file__).resolve().parents[1] / "fixtures" / "results"
    / "elasticFrame.mpco"
)


# ---------------------------------------------------------------------------
# plot.energy
# ---------------------------------------------------------------------------

def test_energy_plot_draws_all_components() -> None:
    r = Results.from_ladruno(ENERGY)
    ax = r.plot.energy()
    assert ax is not None
    # KE / IE / DW / ULW / RES on the energy axis (ERR goes to the twin).
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert labels == ["KE", "IE", "DW", "ULW", "RES"]
    # 5 transient steps in the fixture.
    assert all(ln.get_xdata().size == 5 for ln in ax.get_lines())
    # ERR rides a twin axis sharing x.
    twins = [
        other for other in ax.figure.axes
        if other is not ax and other.get_shared_x_axes().joined(ax, other)
    ]
    assert len(twins) == 1
    assert [ln.get_label() for ln in twins[0].get_lines()] == ["ERR [%]"]


def test_energy_plot_per_region() -> None:
    r = Results.from_ladruno(ENERGY)
    ax = r.plot.energy(region=1)
    assert "region 1" in ax.get_title()


def test_energy_plot_absent_channel_raises() -> None:
    # truss2d was recorded without -G energy — the reader error surfaces.
    r = Results.from_ladruno(TRUSS)
    with pytest.raises(ValueError, match="no ON_DOMAIN/energyBalance"):
        r.plot.energy()


def test_energy_plot_non_ladruno_raises_typeerror() -> None:
    # A committed MPCO file, not the demo: the demo writes a .ladruno
    # since #2778 (ADR 0131 D10), so it has the energy reader seam.
    from tests.conftest import _stub_model_h5_path

    r = Results.from_mpco(MPCO_FRAME, model_h5=_stub_model_h5_path())
    try:
        with pytest.raises(TypeError, match="Ladruno-recorder feature"):
            r.plot.energy()
    finally:
        r.close()


# ---------------------------------------------------------------------------
# plot.node_envelope
# ---------------------------------------------------------------------------

def test_node_envelope_plot_paints_absmax() -> None:
    r = Results.from_ladruno(NODE_ENVELOPE)
    ax = r.plot.node_envelope("displacement_x")
    assert type(ax).__name__ == "Axes3D"
    assert "envelope" in ax.get_title()
    # The painted scalars are the envelope column, not a time-series step:
    # the per-segment values are means of the per-node absmax extremes.
    df = r.node_envelope("displacement_x")
    painted = [
        coll for coll in ax.collections if coll.get_array() is not None
    ]
    assert painted, "no scalar-mapped collection was drawn"
    vals = np.concatenate([
        np.asarray(coll.get_array(), dtype=np.float64) for coll in painted
    ])
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert float(finite.max()) <= float(df["absmax"].max()) + 1e-12


def test_node_envelope_plot_measure_validated() -> None:
    r = Results.from_ladruno(NODE_ENVELOPE)
    with pytest.raises(ValueError, match="measure must be"):
        r.plot.node_envelope("displacement_x", measure="mean")


def test_node_envelope_plot_absent_envelope_raises() -> None:
    # truss2d was recorded as a plain time series (no -envelope).
    r = Results.from_ladruno(TRUSS)
    with pytest.raises(ValueError, match="-envelope"):
        r.plot.node_envelope("displacement_x")
