"""Live: ``DomainCaptureSpec.layers`` on a bridge-attached spec.

``_resolve_layer_section_metadata`` read the legacy ``g.opensees``
registries (``_sections`` / ``_elem_assignments``). The ``apeSees`` bridge
has neither, so a bridge-attached ``layers`` record always resolved to "no
layered-section metadata" and ``_LayerCapturer`` refused it. The lookup
now walks the bridge's ``Element`` primitives and their ``LayeredShell``
sections.

The model is one ``ASDShellQ4`` (1 x 1 in XY) with a four-layer section —
concrete (``PlateFromPlaneStress``), two ``PlateRebar`` bars at 0 and 90 deg,
concrete — sheared in its plane. The quad's ``fem_eid`` sits after gmsh's
point and line elements, so it is NOT the ops tag (1); the capture must
query the ops tag and write the fem id back.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.section.plate import ShellLayer
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

E_C, H_C = 30e9, 0.2
E_S = 200e9
T_BAR = 0.001
V = 1.0e5
THICKNESS = [H_C / 2, T_BAR, T_BAR, H_C / 2]


def _one_quad(g):
    G = g.model.geometry
    pts = [G.add_point(x, y, 0.0) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    lines = [G.add_line(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    surf = G.add_plane_surface([G.add_curve_loop(lines)])
    g.model.sync()
    g.physical.add(2, [surf], name="Wall")
    g.physical.add(1, [lines[0]], name="Base")
    g.physical.add(1, [lines[2]], name="Top")
    st = g.mesh.structured
    for ln in lines:
        st.set_transfinite_curve(ln, 2)
    st.set_transfinite_surface(surf)
    st.set_recombine(surf)
    g.mesh.generation.generate(2)
    return g.mesh.queries.get_fem_data(dim=2)


@pytest.mark.live
def test_layers_capture_resolves_through_the_bridge(g, tmp_path: Path) -> None:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    fem = _one_quad(g)
    (wall_eid,) = [int(e) for e in fem.elements.physical.element_ids("Wall")]
    assert wall_eid != 1       # the fem <-> ops translation is exercised

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=E_S)
    conc = ops.nDMaterial.ElasticIsotropic(E=E_C, nu=0.2)
    conc_layer = ops.nDMaterial.PlateFromPlaneStress(
        material=conc, G_out=E_C / 2.4,
    )
    bar_x = ops.nDMaterial.PlateRebar(material=steel, angle=0.0)
    bar_y = ops.nDMaterial.PlateRebar(material=steel, angle=90.0)
    sec = ops.section.LayeredShell(layers=(
        ShellLayer(material=conc_layer, thickness=THICKNESS[0]),
        ShellLayer(material=bar_x, thickness=THICKNESS[1]),
        ShellLayer(material=bar_y, thickness=THICKNESS[2]),
        ShellLayer(material=conc_layer, thickness=THICKNESS[3]),
    ))
    ops.element.ASDShellQ4(pg="Wall", section=sec)
    # Membrane only: out-of-plane translation and every rotation held.
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.fix(pg="Top", dofs=(0, 0, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Top", forces=(V / 2, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    spec = DomainCaptureSpec(opensees=ops)
    spec.layers(pg="Wall", components=("fiber_stress", "fiber_strain"),
                name="stack")
    meta = spec.resolve(fem).records[0].layer_section_metadata
    assert meta is not None
    assert meta.element_to_section == {wall_eid: ops.tag_for(sec)}
    (sdef,) = meta.sections.values()
    assert sdef.n_layers == 4
    assert list(sdef.material_tags) == [
        ops.tag_for(m) for m in (conc_layer, bar_x, bar_y, conc_layer)
    ]

    path = str(tmp_path / "layers.h5")
    with ops.domain_capture(spec, path=path, ops=emitter.ops) as cap:
        cap.begin_stage("shear", kind="static")
        for _ in range(2):
            assert emitter.analyze(steps=1) == 0
            cap.step(t=emitter.ops.getTime())
        cap.end_stage()

    om = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, model=om) as r:
        layers = r.elements.layers
        # PlateFiber: 5 components per layer cell.
        stress = [layers.get(component=f"fiber_stress_{k}") for k in range(5)]
        strain = [layers.get(component=f"fiber_strain_{k}") for k in range(5)]

    s0 = stress[0]
    # 1 element x 4 surface GPs x 4 layers, 2 steps.
    assert s0.values.shape == (2, 16)
    assert set(int(e) for e in s0.element_index) == {wall_eid}
    np.testing.assert_array_equal(s0.layer_index, [0, 1, 2, 3] * 4)
    np.testing.assert_allclose(s0.thickness, THICKNESS * 4)

    last = s0.layer_index
    conc_rows = (last == 0) | (last == 3)
    bar_rows = (last == 1) | (last == 2)
    # In-plane shear reaches the concrete layers (component 2 = tau_12).
    assert np.all(np.abs(stress[2].values[-1][conc_rows]) > 1e3)
    # A bar carries only its own axial stress, E_s times its strain along
    # the bar, whichever section axis the build puts it on.
    axial = np.where(last[bar_rows] == 1, 0, 1)      # 0 deg -> 11, 90 -> 22
    got = np.array([stress[a].values[-1][bar_rows][i]
                    for i, a in enumerate(axial)])
    eps = np.array([strain[a].values[-1][bar_rows][i]
                    for i, a in enumerate(axial)])
    assert np.any(np.abs(eps) > 0)
    np.testing.assert_allclose(got, E_S * eps, rtol=1e-6, atol=1e-3)
    # The load doubles between the steps, and the response is linear.
    np.testing.assert_allclose(
        stress[2].values[1], 2.0 * stress[2].values[0], rtol=1e-8,
    )
