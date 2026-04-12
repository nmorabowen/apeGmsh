"""
Tests for the ``apeGmsh.sections`` parametric section factories.
"""
from __future__ import annotations

import pytest
import gmsh

from apeGmsh import apeGmsh
from apeGmsh.sections import (
    W_solid, W_shell, W_profile,
    rect_solid, rect_hollow,
    pipe_solid, pipe_hollow,
    angle_solid, channel_solid, tee_solid,
)


class TestWSolid:
    def test_produces_7_hex_volumes(self):
        col = W_solid(bf=150, tf=20, h=300, tw=10, length=3000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(col)
                assert len(gmsh.model.getEntities(3)) == 7
        finally:
            col.cleanup()

    def test_labels_top_web_bottom(self):
        col = W_solid(bf=150, tf=20, h=300, tw=10, length=3000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(col, label="c")
                labels = g.labels.get_all()
                assert "c.top_flange" in labels
                assert "c.bottom_flange" in labels
                assert "c.web" in labels
        finally:
            col.cleanup()

    def test_transfinite_automatic_produces_hexes(self):
        col = W_solid(bf=150, tf=20, h=300, tw=10, length=3000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(col)
                g.mesh.structured.set_transfinite_automatic()
                g.mesh.sizing.set_global_size(80)
                g.mesh.generation.generate(3)
                fem = g.mesh.queries.get_fem_data(dim=3)
                assert fem.connectivity.shape[1] == 8  # hexes
        finally:
            col.cleanup()


class TestWShell:
    def test_produces_3_surfaces(self):
        bm = W_shell(bf=190, tf=14, h=428, tw=9, length=4000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(bm)
                assert len(gmsh.model.getEntities(2)) == 3
        finally:
            bm.cleanup()

    def test_labels_top_web_bottom(self):
        bm = W_shell(bf=190, tf=14, h=428, tw=9, length=4000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(bm, label="b")
                labels = g.labels.get_all()
                assert "b.top_flange" in labels
                assert "b.bottom_flange" in labels
                assert "b.web" in labels
        finally:
            bm.cleanup()


class TestWProfile:
    def test_produces_1_surface_no_volumes(self):
        prof = W_profile(bf=150, tf=20, h=300, tw=10)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(prof)
                assert len(gmsh.model.getEntities(3)) == 0
                assert len(gmsh.model.getEntities(2)) >= 1
        finally:
            prof.cleanup()

    def test_has_profile_label(self):
        prof = W_profile(bf=150, tf=20, h=300, tw=10)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(prof, label="p")
                assert "p.profile" in g.labels.get_all()
        finally:
            prof.cleanup()


class TestRectSolid:
    def test_produces_1_volume(self):
        bar = rect_solid(b=100, h=200, length=5000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(bar)
                assert len(gmsh.model.getEntities(3)) == 1
        finally:
            bar.cleanup()

    def test_has_body_label(self):
        bar = rect_solid(b=100, h=200, length=5000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(bar, label="r")
                assert "r.body" in g.labels.get_all()
        finally:
            bar.cleanup()


class TestRectHollow:
    def test_produces_1_hollow_volume(self):
        hss = rect_hollow(b=200, h=100, t=10, length=3000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(hss)
                assert len(gmsh.model.getEntities(3)) == 1
        finally:
            hss.cleanup()


class TestPipeSolid:
    def test_produces_1_volume(self):
        rod = pipe_solid(r=50, length=2000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(rod)
                assert len(gmsh.model.getEntities(3)) == 1
        finally:
            rod.cleanup()


class TestPipeHollow:
    def test_produces_1_hollow_volume(self):
        tube = pipe_hollow(r_outer=100, t=8, length=4000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(tube)
                assert len(gmsh.model.getEntities(3)) == 1
        finally:
            tube.cleanup()


class TestAngleSolid:
    def test_produces_3_volumes(self):
        ang = angle_solid(b=100, h=100, t=10, length=2000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(ang)
                assert len(gmsh.model.getEntities(3)) == 3
        finally:
            ang.cleanup()

    def test_has_leg_labels(self):
        ang = angle_solid(b=100, h=100, t=10, length=2000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(ang, label="a")
                labels = g.labels.get_all()
                assert "a.horizontal_leg" in labels
                assert "a.vertical_leg" in labels
        finally:
            ang.cleanup()


class TestChannelSolid:
    def test_produces_5_volumes(self):
        ch = channel_solid(bf=80, tf=12, h=200, tw=8, length=3000)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(ch)
                assert len(gmsh.model.getEntities(3)) == 5
        finally:
            ch.cleanup()

    def test_has_flange_web_labels(self):
        ch = channel_solid(bf=80, tf=12, h=200, tw=8, length=3000)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(ch, label="c")
                labels = g.labels.get_all()
                assert "c.top_flange" in labels
                assert "c.bottom_flange" in labels
                assert "c.web" in labels
        finally:
            ch.cleanup()


class TestTeeSolid:
    def test_produces_4_volumes(self):
        t = tee_solid(bf=150, tf=15, h=200, tw=10, length=2500)
        try:
            with apeGmsh(model_name="t") as g:
                g.parts.add(t)
                assert len(gmsh.model.getEntities(3)) == 4
        finally:
            t.cleanup()

    def test_has_flange_stem_labels(self):
        t = tee_solid(bf=150, tf=15, h=200, tw=10, length=2500)
        try:
            with apeGmsh(model_name="t") as g:
                inst = g.parts.add(t, label="t")
                labels = g.labels.get_all()
                assert "t.flange" in labels
                assert "t.stem" in labels
        finally:
            t.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
