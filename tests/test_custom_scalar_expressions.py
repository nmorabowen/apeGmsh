"""User-defined scalar expressions (ADR 0076).

Two layers: the pure expression engine (``_expr``) and the compute-on-read
wiring through the ``results.nodes`` / ``results.elements.gauss`` composites.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results, _expr
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5


# =====================================================================
# Engine — _expr
# =====================================================================

def _ns(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float) for k, v in arrays.items()}


def test_compile_and_evaluate_basic() -> None:
    ns = _ns(vx=[[0.0, 1.0, 2.0]], vy=[[3.0, 4.0, 5.0]])
    d = _expr.compile_expr("k", "vx**2 + vy**2", available={"vx", "vy"})
    assert d.operands == frozenset({"vx", "vy"})
    np.testing.assert_allclose(_expr.evaluate(d, ns), [[9.0, 17.0, 29.0]])


def test_label_units_default_and_override() -> None:
    d = _expr.compile_expr("dcr", "vx / 2", available={"vx"})
    assert d.label == "dcr" and d.units == ""
    d2 = _expr.compile_expr("dcr", "vx / 2", available={"vx"},
                            label="Ratio", units="-")
    assert d2.label == "Ratio" and d2.units == "-"


def test_where_and_bitwise_and_comparison() -> None:
    ns = _ns(a=[[1.0, 5.0, 9.0]], b=[[2.0, 2.0, 2.0]])
    d = _expr.compile_expr("w", "where((a > b) & (a < 9), a, -1)",
                           available={"a", "b"})
    np.testing.assert_allclose(_expr.evaluate(d, ns), [[-1.0, 5.0, -1.0]])


def test_minimum_maximum_are_elementwise() -> None:
    ns = _ns(a=[[1.0, 9.0]], b=[[5.0, 5.0]])
    dmin = _expr.compile_expr("m", "minimum(a, b)", available={"a", "b"})
    dmax = _expr.compile_expr("x", "maximum(a, b)", available={"a", "b"})
    np.testing.assert_allclose(_expr.evaluate(dmin, ns), [[1.0, 5.0]])
    np.testing.assert_allclose(_expr.evaluate(dmax, ns), [[5.0, 9.0]])


@pytest.mark.parametrize("bad", [
    '__import__("os")',        # dunder call
    "a.real",                  # attribute access
    "a[0]",                    # subscript
    "a if b else 0",           # IfExp — banned, not lowered
    "a and b",                 # BoolOp — banned, not lowered
    "a or b",
    "min(a, b)",               # reduction/variadic builtins not exposed
    "max(a)",
    "sum(a)",
    "foo(a)",                  # unknown function
    "1 < a < 3",               # chained comparison
    "not a",                   # unary not
    "lambda: 1",               # lambda
])
def test_disallowed_syntax_rejected(bad: str) -> None:
    with pytest.raises(_expr.ExprError):
        _expr.compile_expr("b", bad, available={"a", "b"})


def test_unknown_operand_rejected() -> None:
    with pytest.raises(_expr.ExprError, match="unknown operand 'zzz'"):
        _expr.compile_expr("b", "a + zzz", available={"a"})


def test_function_arity_enforced() -> None:
    with pytest.raises(_expr.ExprError, match="takes 2 argument"):
        _expr.compile_expr("b", "minimum(a)", available={"a"})
    with pytest.raises(_expr.ExprError, match="takes 1 argument"):
        _expr.compile_expr("b", "sqrt(a, a)", available={"a"})


def test_shape_mismatch_raises_legibly() -> None:
    d = _expr.compile_expr("m", "a + b", available={"a", "b"})
    ns = {"a": np.ones((2, 3)), "b": np.ones((2, 4))}
    with pytest.raises(_expr.ExprError, match="different points"):
        _expr.evaluate(d, ns)


def test_bool_literal_rejected() -> None:
    with pytest.raises(_expr.ExprError, match="numeric literal"):
        _expr.compile_expr("b", "a + True", available={"a"})


# =====================================================================
# Composite wiring — nodes
# =====================================================================

def _write_nodes(path: Path, *, comps: dict[str, np.ndarray],
                 time: np.ndarray, node_ids: np.ndarray) -> None:
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids, components=comps)
        w.end_stage()


def _node_results(tmp_path: Path) -> Path:
    path = tmp_path / "nodes.h5"
    time = np.array([0.0, 1.0])
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    comps = {
        "velocity_x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "velocity_y": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        "displacement_x": np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]),
    }
    _write_nodes(path, comps=comps, time=time, node_ids=node_ids)
    return path


def test_node_define_get_and_available(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        n = r.nodes
        n.define("kinetic_ish", "velocity_x**2 + velocity_y**2 + displacement_x")
        assert "kinetic_ish" in n.available_components()
        assert "kinetic_ish" in n.definitions

        vx = n.get(component="velocity_x").values
        vy = n.get(component="velocity_y").values
        dx = n.get(component="displacement_x").values
        got = n.get(component="kinetic_ish")
        assert got.component == "kinetic_ish"
        np.testing.assert_array_equal(got.node_ids, [1, 2, 3])
        np.testing.assert_allclose(got.values, vx**2 + vy**2 + dx)


def test_node_custom_respects_selection(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        n = r.nodes
        n.define("d2", "displacement_x * 2")
        sub = n.get(component="d2", ids=[2])
        np.testing.assert_array_equal(sub.node_ids, [2])
        np.testing.assert_allclose(sub.values, [[20.0], [40.0]])


def test_custom_on_custom(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        n = r.nodes
        n.define("a", "velocity_x + 1")
        n.define("b", "a * 10")               # references the earlier custom
        vx = n.get(component="velocity_x").values
        np.testing.assert_allclose(n.get(component="b").values, (vx + 1) * 10)


def test_shadow_and_redefine_refused(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        n = r.nodes
        with pytest.raises(ValueError, match="shadows"):
            n.define("velocity_x", "velocity_y * 2")
        n.define("foo", "velocity_x")
        with pytest.raises(ValueError, match="already defined"):
            n.define("foo", "velocity_y")


def test_undefine_and_dependency_guard(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        n = r.nodes
        n.define("a", "velocity_x + 1")
        n.define("b", "a * 10")
        with pytest.raises(ValueError, match="depend on it"):
            n.undefine("a")
        n.undefine("b")
        n.undefine("a")                       # now free to remove
        assert "a" not in n.definitions
        with pytest.raises(KeyError):
            n.undefine("nope")


def test_zero_operand_expression_refused(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        with pytest.raises(ValueError, match="no recorded component"):
            r.nodes.define("const", "1 + 2")


# =====================================================================
# Composite wiring — gauss, composing on a derived scalar
# =====================================================================

_STRESS = ("xx", "yy", "zz", "xy", "yz", "xz")


def _gauss_results(tmp_path: Path) -> Path:
    path = tmp_path / "gauss.h5"
    time = np.array([0.0, 1.0, 2.0])
    elem_idx = np.array([10, 20], dtype=np.int64)
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    rng = np.random.default_rng(7)
    comps = {f"stress_{s}": rng.standard_normal((3, 2, 1)) for s in _STRESS}
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_gauss_group(
            sid, "partition_0", "group_0", class_tag=4, int_rule=1,
            element_index=elem_idx, natural_coords=nat, components=comps,
        )
        w.end_stage()
    return path


def test_gauss_custom_composes_on_derived(tmp_path: Path) -> None:
    path = _gauss_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        g.define("dcr", "von_mises_stress / 250.0", units="-")
        assert "dcr" in g.available_components()
        vm = g.get(component="von_mises_stress").values
        dcr = g.get(component="dcr")
        assert dcr.component == "dcr"
        np.testing.assert_array_equal(dcr.element_index, [10, 20])
        np.testing.assert_allclose(dcr.values, vm / 250.0)


def test_gauss_and_node_registries_are_independent(tmp_path: Path) -> None:
    path = _gauss_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        r.elements.gauss.define("dcr", "von_mises_stress / 250.0")
        # A gauss-domain name is not visible on the node composite.
        assert "dcr" not in r.nodes.definitions


# =====================================================================
# Viewer reach (Slice 3) — cross-process transport
# =====================================================================

def test_definitions_payload_roundtrip(tmp_path: Path) -> None:
    import json

    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        r.nodes.define("a", "velocity_x + 1")
        r.nodes.define("b", "a * 10", label="B", units="m/s")   # depends on a
        payload = r._definitions_payload()
        # Definition order preserved so 'a' replays before 'b'.
        assert [d["name"] for d in payload] == ["a", "b"]
        assert payload[1]["domain"] == "node"
        wire = json.loads(json.dumps(payload))    # survives JSON

    # Fresh Results over the same file, as the subprocess would open it.
    with Results.from_native(path, model=_open_model_from_h5(path)) as r2:
        r2._apply_definitions_payload(wire)
        assert "b" in r2.nodes.available_components()
        vx = r2.nodes.get(component="velocity_x").values
        np.testing.assert_allclose(
            r2.nodes.get(component="b").values, (vx + 1) * 10,
        )
        assert r2.nodes.definitions["b"].units == "m/s"


def test_gauss_payload_roundtrip(tmp_path: Path) -> None:
    path = _gauss_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        r.elements.gauss.define("dcr", "von_mises_stress / 250.0", units="-")
        payload = r._definitions_payload()
        assert payload[0]["domain"] == "gauss"
    with Results.from_native(path, model=_open_model_from_h5(path)) as r2:
        r2._apply_definitions_payload(payload)
        vm = r2.elements.gauss.get(component="von_mises_stress").values
        np.testing.assert_allclose(
            r2.elements.gauss.get(component="dcr").values, vm / 250.0,
        )


def test_viewer_argv_carries_defs_flag(tmp_path: Path) -> None:
    path = _node_results(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        # No defs → no --defs.
        assert "--defs" not in r._build_viewer_argv(title=None)
        # With a defs path → flag present and points at the sidecar.
        dp = r._default_defs_path(path)
        argv = r._build_viewer_argv(title=None, defs_path=dp)
        assert "--defs" in argv
        assert str(dp) in argv
        assert str(dp).endswith(".defs.json")


def _patch_opener(monkeypatch):
    """Route __main__._open_results through the stub-tolerant opener so a
    synthetic (model-less) NativeWriter file can boot the viewer path."""
    from apeGmsh.viewers import __main__ as vm

    def _open(path, model_h5):
        return Results.from_native(path, model=_open_model_from_h5(path))

    monkeypatch.setattr(vm, "_open_results", _open)


def test_main_applies_defs_under_skip_viewer(tmp_path: Path, monkeypatch) -> None:
    """__main__ loads --defs and applies them before opening the (skipped)
    viewer, without error."""
    import json

    from apeGmsh.viewers import __main__ as vm

    path = _node_results(tmp_path)
    defs = tmp_path / "nodes.h5.defs.json"
    defs.write_text(json.dumps(
        [{"name": "kx", "expr": "velocity_x * 2", "domain": "node",
          "label": "kx", "units": ""}]
    ), encoding="utf-8")

    # Capture the Results the viewer would have received.
    seen = {}
    real_apply = vm._apply_defs

    def _spy(results, defs_path):
        real_apply(results, defs_path)
        seen["kx"] = "kx" in results.nodes.definitions

    _patch_opener(monkeypatch)
    monkeypatch.setattr(vm, "_apply_defs", _spy)
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")

    assert vm.main([str(path), "--defs", str(defs)]) == 0
    assert seen["kx"] is True


def test_main_survives_malformed_defs(tmp_path: Path, monkeypatch, capsys) -> None:
    from apeGmsh.viewers import __main__ as vm

    path = _node_results(tmp_path)
    defs = tmp_path / "bad.defs.json"
    defs.write_text("{not json", encoding="utf-8")

    _patch_opener(monkeypatch)
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    assert vm.main([str(path), "--defs", str(defs)]) == 0
    assert "could not apply custom scalar definitions" in capsys.readouterr().err
