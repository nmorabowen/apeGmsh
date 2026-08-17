"""ADR 0098 S2 — acceptance-pass regressions: pose sync + true luts.

The first human pass over the session window found two realize-layer
defects this file pins:

1. **Glyph kinds ignored the pose.** Warp-before-extract (S1 decision
   5) only poses kinds that extract from ``scene.grid``; kinds that
   OWN their glyph geometry (gauss markers, vector/principal anchors)
   place it from the REFERENCE view data at attach. The old viewer
   moves them with the DEFORM pump's ``sync_substrate_points``; the
   session path now runs that primitive one-shot after attach.
2. **Colour-mapped glyph/sand layers painted one colour.** Those kinds
   never compute a lookup-table range — the emitted ``LutSpec`` is the
   placeholder ``(0, 1)`` (or ``principal_glyph``'s signed ``(-1, 1)``
   fallback), so on the session path the actor clamps every value to
   the top colour and the bar reads the placeholder. The old window
   re-ranges through the Qt LUT machinery, which the null legend
   controller (decision 4) deliberately silences. Realize now ranges
   the placeholder from the data (symmetrically for the signed case)
   and the bar follows the same corrected record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, Deform, Gauss, Sand, Vector
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"
N_STEPS = 3


@pytest.fixture
def session_results(g, tmp_path: Path):
    """Cube with UNAMBIGUOUS data: displacement_z = nid + t*1000 (posed
    coordinates leave the unit cube by orders of magnitude) and 1-GP
    gauss stress_xx = eid*10 + t (range far outside any placeholder
    lut)."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = np.concatenate(
        [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    )
    disp = np.zeros((N_STEPS, node_ids.size))
    for t in range(N_STEPS):
        disp[t] = node_ids + t * 1000.0
    sxx = np.zeros((N_STEPS, elem_ids.size, 1))
    for t in range(N_STEPS):
        sxx[t, :, 0] = elem_ids * 10.0 + t

    path = tmp_path / "session_pose_lut.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=np.zeros((1, 3)),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _realize(results, **slots):
    session = results.session()
    view = session.panes[0]
    for name, record in slots.items():
        setattr(view, name, record)
    backend = RecordingBackend()
    realized = session.realize(backend=backend)
    return view, backend, realized


def _slot_layer(backend, realized, category):
    (entry,) = [
        lyr for lyr in realized.layers
        if lyr.key.endswith(f":{category}")
    ]
    return backend.layers[entry.layer_id]


def _slot_color(backend, realized, category):
    """The EFFECTIVE colour of a slot's layer — what the backend last
    applied (``set_layer_color`` re-ranges after emission, so the
    record in ``backend.layers`` may carry the stale attach-time
    lut; ``backend.colors`` is the actor state)."""
    (entry,) = [
        lyr for lyr in realized.layers
        if lyr.key.endswith(f":{category}")
    ]
    return backend.colors[entry.layer_id]


# =====================================================================
# 1 — glyph kinds follow the pose
# =====================================================================


def test_gauss_glyphs_follow_the_pose(session_results):
    """Deform on → the GP markers ride the posed substrate, not the
    reference mesh (the acceptance defect: spheres floating on the
    undeformed geometry while the mesh warped away)."""
    _v, _b, flat = _realize(session_results, gauss=Gauss("stress_xx"))
    _v, _b, posed = _realize(
        session_results,
        gauss=Gauss("stress_xx"),
        deform=Deform("displacement", scale=1.0),
    )
    flat_z = np.asarray(
        flat.diagrams[0]._layer.positions.coords  # noqa: SLF001
    )[:, 2]
    posed_z = np.asarray(
        posed.diagrams[0]._layer.positions.coords  # noqa: SLF001
    )[:, 2]
    # Reference cube: z within [0, 1]. Posed: displacement_z ≥ 1000·2
    # at the default (last) step, so the glyphs must leave the cube.
    assert float(flat_z.max()) <= 1.0 + 1e-9
    assert float(posed_z.max()) > 1000.0


def test_vector_glyphs_follow_the_pose(session_results):
    _v, _b, posed = _realize(
        session_results,
        vector=Vector("displacement_z"),
        deform=Deform("displacement", scale=1.0),
    )
    anchors_z = np.asarray(
        posed.diagrams[0]._layer.positions.coords  # noqa: SLF001
    )[:, 2]
    assert float(anchors_z.max()) > 1000.0


# =====================================================================
# 2 — placeholder luts are ranged from the data (actor AND bar)
# =====================================================================


def test_gauss_lut_ranges_from_the_data(session_results):
    view, backend, realized = _realize(
        session_results, gauss=Gauss("stress_xx"),
    )
    layer = _slot_layer(backend, realized, "gauss")
    values = np.asarray(layer.color_scalar)
    lut = _slot_color(backend, realized, "gauss").lut
    assert (lut.vmin, lut.vmax) == (
        pytest.approx(float(values.min())),
        pytest.approx(float(values.max())),
    )
    # Far outside the placeholder — the assertion cannot pass by luck.
    assert lut.vmax > 100.0
    # The bar reads the SAME corrected range (the acceptance symptom
    # was both: one-colour picture and a 0–1 bar).
    (spec,) = backend.scalar_bars.values()
    assert (spec.lut.vmin, spec.lut.vmax) == (lut.vmin, lut.vmax)


def test_principal_lut_is_symmetric_about_zero(session_results):
    """The signed (-1, 1) fallback re-ranges SYMMETRICALLY so the
    diverging colormap stays centred (tension/compression law)."""
    view, backend, realized = _realize(
        session_results, vector=Vector("stress_xx"),
    )
    layer = _slot_layer(backend, realized, "vector")
    span = float(np.max(np.abs(np.asarray(layer.color_scalar))))
    lut = _slot_color(backend, realized, "vector").lut
    assert lut.vmin == pytest.approx(-span)
    assert lut.vmax == pytest.approx(span)
    assert span > 100.0


def test_sand_lut_ranges_from_the_nodal_field(session_results):
    view, backend, realized = _realize(
        session_results, sand=Sand("displacement_z"),
    )
    layer = _slot_layer(backend, realized, "sand")
    field = next(
        f for f in layer.fields if f.name == layer.color.array_name
    )
    lut = _slot_color(backend, realized, "sand").lut
    assert lut.vmax == pytest.approx(float(np.max(field.values)))
    assert lut.vmax > 1000.0  # nid + t*1000 — nowhere near (0, 1)


def test_contour_whole_history_clim_is_left_alone(session_results):
    """The complement guard: the contour computes its own whole-history
    clim (S1-A P10) — the auto-fit correction must not narrow it to
    the displayed step. Whole-history means the SAME lut at every
    instant; a stomped per-step lut would differ between step 0 and
    the last step (the data shifts by +t)."""
    from apeGmsh.results.session import Instant

    session = session_results.session()
    view = session.panes[0]
    view.contour = Contour("stress_xx")

    luts = []
    for step in (0, N_STEPS - 1):
        session.time = Instant(STAGE, step)
        backend = RecordingBackend()
        realized = session.realize(backend=backend)
        lut = _slot_layer(backend, realized, "contour").color.lut
        luts.append((lut.vmin, lut.vmax))
    assert luts[0] == luts[1]
