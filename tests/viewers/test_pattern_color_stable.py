"""A load pattern gets the same colour in every session.

``loads_tab.pattern_color`` colours a pattern's arrows in the mesh
viewer and its label in the Loads tab. It used ``abs(hash(name))``,
which is randomized per process (``PYTHONHASHSEED``), so the colours
changed between sessions. A name digest alone is no cure either: in
seven slots ``dead`` and ``live`` collide under crc32, and five patterns
collide about 85 % of the time under any digest.

A declared pattern now takes the palette slot of its index in
``g.loads.cases()`` (declaration order). Both callers pass that order;
neither may use ``view.nodes.loads``, which ``FEMData.from_h5`` reads
back alphabetically. A name that is not declared falls back to
``zlib.crc32`` of the name (#374 / 184b5734), checked across processes
below: within one process ``hash()`` is constant, so an in-process
check would pass on the broken code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apeGmsh.viewers.ui.loads_tab import pattern_color

# Palette slots 0, 1 and 2, written out so a reshuffle is noticed.
GREEN, PEACH, YELLOW = "#a6e3a1", "#fab387", "#f9e2af"
DECLARED = ["live", "dead", "wind"]  # not alphabetical, on purpose

_SRC = Path(__file__).resolve().parents[2] / "src"
_NAMES = ("dead", "live", "wind", "snow", "EQ_x", "EQ_y", "pattern_1", "Pattern 2")

# Pure: loads_tab imports Qt only inside _qt(), so no display is needed.
_PROBE = (
    "import json, apeGmsh.viewers.ui.loads_tab as m\n"
    f"print(json.dumps([m.__file__, [m.pattern_color(n) for n in {_NAMES!r}]]))\n"
)


def test_declared_patterns_take_slots_in_declaration_order() -> None:
    assert pattern_color("live", DECLARED) == GREEN
    assert pattern_color("dead", DECLARED) == PEACH
    assert pattern_color("wind", DECLARED) == YELLOW
    # The eighth declared pattern wraps around to the first slot.
    eight = [f"p{i}" for i in range(8)]
    assert pattern_color("p7", eight) == GREEN


@pytest.mark.parametrize(
    "cases",
    [["dead", "live"], ["live", "dead"], ["Dead", "Live"], ["dead", "live", "L"]],
)
def test_declared_patterns_do_not_collide(cases: list[str]) -> None:
    # Each of these sets shares ONE slot under crc32 % 7.
    colors = [pattern_color(n, cases) for n in cases]
    assert len(set(colors)) == len(colors), dict(zip(cases, colors))


def test_undeclared_name_falls_back_to_crc32() -> None:
    # crc32("snow") % 7 == 0 and crc32("wind") % 7 == 1.
    assert pattern_color("snow") == GREEN
    assert pattern_color("wind", ["dead"]) == PEACH


def _colors_under_seed(seed: str) -> list[str]:
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(_SRC))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env, capture_output=True, text=True, timeout=120, check=True,
    )
    module_file, colors = json.loads(out.stdout.strip().splitlines()[-1])
    # A stale editable install would test some other tree's loads_tab.
    assert Path(module_file).resolve().is_relative_to(_SRC.resolve()), module_file
    return colors


def test_fallback_is_the_same_across_hash_seeds() -> None:
    assert _colors_under_seed("1") == _colors_under_seed("2")


def test_cases_is_declaration_order_in_a_real_session() -> None:
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="pattern_order") as g:
        for i, name in enumerate(DECLARED):
            g.model.geometry.add_point(float(i), 0.0, 0.0, label=f"n{i}")
            with g.loads.case(name):
                g.loads.point.force(f"n{i}", (0.0, 0.0, -1.0))
        cases = g.loads.cases()
    assert cases == DECLARED
    assert [pattern_color(n, cases) for n in DECLARED] == [GREEN, PEACH, YELLOW]


class _Plotter:
    def __init__(self) -> None:
        self.colors: dict[str, str] = {}

    def add_mesh(self, mesh, *, color, name, **kwargs):
        self.colors[name] = color
        return name

    def remove_actor(self, actor) -> None:
        pass


def test_overlay_arrows_follow_cases_not_the_view_order() -> None:
    pytest.importorskip("pyvista")
    import numpy as np

    from apeGmsh.viewers.mesh_viewer import MeshViewer

    mv = MeshViewer.__new__(MeshViewer)
    mv._plotter = _Plotter()
    mv._registry = SimpleNamespace(origin_shift=np.zeros(3), dim_meshes={})
    mv._load_actors = []
    mv._overlay_model = SimpleNamespace(scale=lambda key: 1.0)
    mv._parent = SimpleNamespace(loads=SimpleNamespace(cases=lambda: list(DECLARED)))
    # The view lists patterns alphabetically, as FEMData.from_h5 reads them.
    records = [
        SimpleNamespace(pattern=p, node_id=1, force_xyz=(0.0, 0.0, -1.0), moment_xyz=None)
        for p in sorted(DECLARED)
    ]
    mv._view = SimpleNamespace(
        nodes=SimpleNamespace(loads=records, coords=np.zeros((1, 3)), index=lambda nid: 0),
    )

    mv._rebuild_loads_overlay(set(DECLARED))

    assert mv._plotter.colors == {
        "_loads_force_live": GREEN,
        "_loads_force_dead": PEACH,
        "_loads_force_wind": YELLOW,
    }


def test_loads_tab_labels_follow_cases() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from apeGmsh._kernel.defs.loads import PointLoadDef
    from apeGmsh.viewers.ui.loads_tab import LoadsTabPanel

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    defs = [
        PointLoadDef(target=f"n{i}", pattern=p, force_xyz=(0.0, 0.0, -1.0))
        for i, p in enumerate(DECLARED)
    ]
    tab = LoadsTabPanel(SimpleNamespace(cases=lambda: list(DECLARED), load_defs=defs))

    shown = {p: tab._pattern_items[p].foreground(0).color().name() for p in DECLARED}
    assert shown == {"live": GREEN, "dead": PEACH, "wind": YELLOW}
