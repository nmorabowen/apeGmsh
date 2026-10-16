"""Phase 3 — STKO ↔ canonical name translation.

Also #1946: the modal groups ``MODES_OF_VIBRATION(U|R)`` read back as
``kind="mode"`` stages on both the MPCO and the ``.ladruno`` reader.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh.results._bind import StageCountMismatchWarning, _bind_stage_names
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.readers._mpco_translation import (
    all_known_mpco_result_names,
    canonical_node_component,
    canonical_to_mpco_lookup,
    component_axis,
    has_canonical_mapping,
)
from apeGmsh.results.readers._protocol import ResultLevel
from apeGmsh.results.schema._versions import LADRUNO_SUPPORTED_FORMAT_VERSIONS


# =====================================================================
# Component axis extraction
# =====================================================================

def test_component_axis_latin() -> None:
    assert component_axis("Ux") == "x"
    assert component_axis("Uy") == "y"
    assert component_axis("Uz") == "z"
    assert component_axis("Rx") == "x"
    assert component_axis("Vy") == "y"
    assert component_axis("Az") == "z"
    assert component_axis("RFx") == "x"
    assert component_axis("RMy") == "y"
    assert component_axis("RVz") == "z"
    assert component_axis("RAx") == "x"


def test_component_axis_case_insensitive() -> None:
    assert component_axis("UX") == "x"
    assert component_axis("uX") == "x"


def test_component_axis_returns_none_for_scalar() -> None:
    # Pressure component label is just "p"
    assert component_axis("p") is None
    assert component_axis("P") is None
    assert component_axis("") is None


def test_component_axis_unicode() -> None:
    """Greek-prefixed variants from the docs (ω, α) still end in x/y/z."""
    assert component_axis("ωx") == "x"
    assert component_axis("αy") == "y"


# =====================================================================
# Full translation
# =====================================================================

@pytest.mark.parametrize("mpco_name,comp,expected", [
    ("DISPLACEMENT", "Ux", "displacement_x"),
    ("DISPLACEMENT", "Uy", "displacement_y"),
    ("DISPLACEMENT", "Uz", "displacement_z"),
    ("ROTATION", "Rx", "rotation_x"),
    ("ROTATION", "Ry", "rotation_y"),
    ("ROTATION", "Rz", "rotation_z"),
    ("VELOCITY", "Vx", "velocity_x"),
    ("ANGULAR_VELOCITY", "RVx", "angular_velocity_x"),
    ("ACCELERATION", "Ax", "acceleration_x"),
    ("ANGULAR_ACCELERATION", "RAx", "angular_acceleration_x"),
    ("REACTION_FORCE", "RFx", "reaction_force_x"),
    ("REACTION_MOMENT", "RMz", "reaction_moment_z"),
    ("REACTION_FORCE_INCLUDING_INERTIA", "RFy", "reaction_force_y"),
    # A Rayleigh group is the damping force alone, not a reaction (#2122).
    ("RAYLEIGH_FORCE", "RFx", None),
    ("RAYLEIGH_MOMENT", "RMz", None),
    ("UNBALANCED_FORCE", "Fx", "force_x"),
    ("UNBALANCED_MOMENT", "Mz", "moment_z"),
    ("PRESSURE", "p", "pore_pressure"),
])
def test_canonical_node_component(
    mpco_name: str, comp: str, expected: "str | None",
) -> None:
    assert canonical_node_component(mpco_name, comp) == expected


def test_unknown_mpco_name_returns_none() -> None:
    assert canonical_node_component("NOT_A_RESULT", "Ux") is None


def test_unknown_component_returns_none() -> None:
    # 'Q' isn't a known axis suffix
    assert canonical_node_component("DISPLACEMENT", "Q") is None


# =====================================================================
# Reverse lookup
# =====================================================================

def test_canonical_to_mpco_lookup() -> None:
    assert canonical_to_mpco_lookup("displacement_x") == ("DISPLACEMENT", "x")
    assert canonical_to_mpco_lookup("rotation_z") == ("ROTATION", "z")
    assert canonical_to_mpco_lookup("velocity_y") == ("VELOCITY", "y")


def test_canonical_to_mpco_pressure() -> None:
    assert canonical_to_mpco_lookup("pore_pressure") == ("PRESSURE", "")


def test_canonical_to_mpco_unknown() -> None:
    assert canonical_to_mpco_lookup("not_a_name") is None


# =====================================================================
# Coverage
# =====================================================================

def test_known_mpco_names_includes_all_real_world_groups() -> None:
    """Sanity: every result-group name we saw in real MPCO files is mapped."""
    real_world = [
        "DISPLACEMENT", "ROTATION", "VELOCITY", "ANGULAR_VELOCITY",
        "ACCELERATION", "ANGULAR_ACCELERATION",
        "REACTION_FORCE", "REACTION_MOMENT",
        "REACTION_FORCE_INCLUDING_INERTIA",
        "REACTION_MOMENT_INCLUDING_INERTIA",
        "UNBALANCED_FORCE", "UNBALANCED_MOMENT",
        "UNBALANCED_FORCE_INCLUDING_INERTIA",
        "UNBALANCED_MOMENT_INCLUDING_INERTIA",
        "PRESSURE",
    ]
    known = set(all_known_mpco_result_names())
    missing = [n for n in real_world if n not in known]
    assert not missing, f"Unmapped MPCO result names: {missing}"
    # The Rayleigh groups are real too, and deliberately unmapped (#2122).
    assert not known.intersection({"RAYLEIGH_FORCE", "RAYLEIGH_MOMENT"})


def test_has_canonical_mapping() -> None:
    assert has_canonical_mapping("DISPLACEMENT")
    assert not has_canonical_mapping("MODES_OF_VIBRATION(U)")  # special case


# =====================================================================
# Modes of vibration (#1946): modal groups read as kind="mode" stages
# =====================================================================
#
# The layout below is the one fork build 79e0623670 writes for both
# ``recorder mpco`` and ``recorder ladruno`` (measured; see
# ``_mpco_nodal_io``): ``ON_NODES/MODES_OF_VIBRATION(U|R)/{ID,
# DATA/STEP_<s>/MODE_<k>}`` with LAMBDA on each mode. The oracle is the
# capture_modes rule: f = sqrt(lambda) / 2 pi, T = 1 / f.

_IDS = np.array([1, 2, 3, 4], dtype=np.int64)
_LAMBDAS = (110.0, 4400.0)
_TIMES = (0.5, 1.0, 0.0)          # two static steps, then the eigen record


def _shape(k: int, n_comp: int) -> np.ndarray:
    """Entry-unique (nNodes, nComp) shape for mode ``k`` (0-based)."""
    return (
        100.0 * (k + 1) + _IDS[:, None].astype(float)
        + 0.1 * np.arange(n_comp)[None, :]
    )


def _write_stage(
    stage, *, mpco: bool, modes_step: int = 2,
    lambdas=_LAMBDAS, rotational: bool = True, drop_lambda: bool = False,
    rot_lambdas=None,
) -> None:
    on = stage.create_group("RESULTS/ON_NODES")
    stage.create_group("RESULTS/ON_ELEMENTS")
    disp = on.create_group("DISPLACEMENT")
    disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
    disp.create_dataset("ID", data=_IDS.reshape(-1, 1))
    if mpco:
        data = disp.create_group("DATA")
        for s, t in enumerate(_TIMES):
            ds = data.create_dataset(f"STEP_{s}", data=np.full((4, 2), s + 1.0))
            ds.attrs["TIME"] = np.array([t])
    else:
        disp.create_dataset("DATA", data=np.ones((len(_TIMES), 4, 2)))
        disp.create_dataset("TIME", data=np.array(_TIMES))
        disp.create_dataset("STEP", data=np.arange(len(_TIMES)))
    groups = [("MODES_OF_VIBRATION(U)", b"Ux,Uy", 2, lambdas)]
    if rotational:
        groups.append(
            ("MODES_OF_VIBRATION(R)", b"Rz", 1, rot_lambdas or lambdas),
        )
    for name, comps, n_comp, lams in groups:
        g = on.create_group(name)
        g.attrs["COMPONENTS"] = np.array([comps])
        g.create_dataset("ID", data=_IDS.reshape(-1, 1))
        step = g.create_group(f"DATA/STEP_{modes_step}")
        step.attrs["TIME"] = np.array([0.0])
        for k, lam in enumerate(lams):
            ds = step.create_dataset(f"MODE_{k}", data=_shape(k, n_comp))
            ds.attrs["MODE"] = np.array([k], dtype=np.int32)
            if not drop_lambda:
                ds.attrs["LAMBDA"] = np.array([lam])


def _write_mpco(path: Path, **kw) -> Path:
    with h5py.File(path, "w") as f:
        f.create_group("INFO")
        _write_stage(f.create_group("MODEL_STAGE[1]"), mpco=True, **kw)
    return path


def _write_ladruno(path: Path, **kw) -> Path:
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = max(LADRUNO_SUPPORTED_FORMAT_VERSIONS)
        stage = f.create_group("MODEL_STAGE[1]")
        # The recorder promotes the stage to KIND=eigen whenever modes
        # are requested, though it holds static steps too.
        stage.attrs["KIND"] = np.bytes_(b"eigen")
        _write_stage(stage, mpco=False, **kw)
    return path


def test_regression_ladruno_kind_eigen_is_not_a_mode(tmp_path: Path) -> None:
    """Finding 1: the recorder writes KIND=eigen on EVERY stage once modes
    are requested, so a gravity stage was listed as a mode with empty
    eigen fields (a silent wrong answer). It is a static stage."""
    with LadrunoReader(_write_ladruno(tmp_path / "m.ladruno")) as r:
        host = r.stages()[0]
    assert host.name == "MODEL_STAGE[1]"
    assert host.kind == "static"
    assert host.eigenvalue is None


def test_regression_mpco_modes_beside_steps_do_not_raise(tmp_path: Path) -> None:
    """Finding 2: the modal group's one STEP was counted against the
    stage's three DISPLACEMENT steps and raised a step-count mismatch."""
    with MPCOReader(_write_mpco(tmp_path / "m.mpco")) as r:
        host = r.stages()[0]
        disp = r.read_nodes(host.id, "displacement_x")
    assert host.n_steps == len(_TIMES)
    np.testing.assert_array_equal(disp.time, _TIMES)


def test_modal_component_translation() -> None:
    from apeGmsh.results.readers._mpco_translation import (
        canonical_modal_component,
        is_modal_result_name,
    )

    assert canonical_modal_component("MODES_OF_VIBRATION(U)", "Ux") == "displacement_x"
    assert canonical_modal_component("MODES_OF_VIBRATION(U)", "Uz") == "displacement_z"
    assert canonical_modal_component("MODES_OF_VIBRATION(R)", "Rz") == "rotation_z"
    assert canonical_modal_component("DISPLACEMENT", "Ux") is None
    assert is_modal_result_name("MODES_OF_VIBRATION(R)")
    assert not is_modal_result_name("DISPLACEMENT")
    # A mode shape is never a displacement history of its host stage.
    assert canonical_node_component("MODES_OF_VIBRATION(U)", "Ux") is None


def _assert_modes(reader) -> None:
    stages = reader.stages()
    host, modes = stages[0], [s for s in stages if s.kind == "mode"]
    assert host.kind in ("static", "transient")
    assert host.n_steps == len(_TIMES)
    assert [s.name for s in modes] == ["mode_1", "mode_2"]
    assert [s.mode_index for s in modes] == [1, 2]
    for k, (s, lam) in enumerate(zip(modes, _LAMBDAS)):
        f_hz = math.sqrt(lam) / (2.0 * math.pi)
        assert s.eigenvalue == lam
        assert s.frequency_hz == pytest.approx(f_hz, rel=1e-14)
        assert s.period_s == pytest.approx(1.0 / f_hz, rel=1e-14)
        assert s.n_steps == 1
        np.testing.assert_array_equal(reader.time_vector(s.id), [0.0])
        assert reader.available_components(s.id, ResultLevel.NODES) == [
            "displacement_x", "displacement_y", "rotation_z",
        ]
        assert reader.available_components(s.id, ResultLevel.GAUSS) == []
        for comp, src, col in (
            ("displacement_x", 2, 0), ("displacement_y", 2, 1),
            ("rotation_z", 1, 0),
        ):
            slab = reader.read_nodes(s.id, comp)
            np.testing.assert_array_equal(slab.node_ids, _IDS)
            np.testing.assert_array_equal(
                slab.values, _shape(k, src)[:, col][None, :],
            )
        sub = reader.read_nodes(s.id, "displacement_y", node_ids=np.array([3]))
        np.testing.assert_array_equal(sub.values, [[_shape(k, 2)[2, 1]]])
        assert reader.node_partition_reduction(s.id, "displacement_x") == "NONE"
    # The host still reads its own displacement history, not a mode.
    host_slab = reader.read_nodes(host.id, "displacement_x")
    assert host_slab.values.shape == (len(_TIMES), _IDS.size)


def test_mpco_reads_modes_as_mode_stages(tmp_path: Path) -> None:
    # Before #1946 this file raised: the modal group's single STEP was
    # counted against the stage's three DISPLACEMENT steps.
    with MPCOReader(_write_mpco(tmp_path / "m.mpco")) as r:
        _assert_modes(r)
        mode = r.stages()[1].id
        assert r.read_gauss(mode, "stress_xx").values.size == 0


def test_ladruno_reads_modes_as_mode_stages(tmp_path: Path) -> None:
    with LadrunoReader(_write_ladruno(tmp_path / "m.ladruno")) as r:
        _assert_modes(r)
        assert r.stages()[0].kind == "static"   # KIND=eigen is not a mode
        mode = r.stages()[1].id
        with pytest.raises(ValueError, match="mode stage"):
            r.read_gauss(mode, "stress_xx")


@pytest.mark.parametrize("ladruno", [False, True])
def test_mode_without_lambda_is_refused(tmp_path: Path, ladruno: bool) -> None:
    if ladruno:
        reader = LadrunoReader(_write_ladruno(tmp_path / "m.ladruno", drop_lambda=True))
    else:
        reader = MPCOReader(_write_mpco(tmp_path / "m.mpco", drop_lambda=True))
    with reader, pytest.raises(ValueError, match="LAMBDA"):
        reader.stages()


def test_inconsistent_rotational_modes_are_refused(tmp_path: Path) -> None:
    path = _write_mpco(tmp_path / "m.mpco", rot_lambdas=(110.0, 4401.0))
    with MPCOReader(path) as r, pytest.raises(ValueError, match="different modes"):
        r.stages()


def test_non_positive_eigenvalue_warns_and_reads_zero(tmp_path: Path) -> None:
    path = _write_mpco(tmp_path / "m.mpco", lambdas=(-1e-6, 4400.0))
    with MPCOReader(path) as r:
        with pytest.warns(UserWarning, match="non-positive eigenvalue"):
            modes = [s for s in r.stages() if s.kind == "mode"]
    assert modes[0].eigenvalue == -1e-6
    assert (modes[0].frequency_hz, modes[0].period_s) == (0.0, 0.0)


def _two_parts(tmp_path: Path, ladruno: bool, **kw) -> "list[Path]":
    ext = "ladruno" if ladruno else "mpco"
    writer = _write_ladruno if ladruno else _write_mpco
    paths = [writer(tmp_path / f"r.part-{i}.{ext}", **kw) for i in range(2)]
    if ladruno:
        for i, p in enumerate(paths):
            with h5py.File(p, "a") as f:
                f["INFO"].attrs["PARTITIONED"] = 1
                f["INFO"].attrs["PARTITION_ID"] = i
                f["INFO"].attrs["NUM_PARTITIONS"] = 2
    return paths


@pytest.mark.parametrize("ladruno", [False, True])
def test_part_set_warns_once_per_non_positive_mode(
    tmp_path: Path, ladruno: bool,
) -> None:
    """Every part sees the same eigen solution: the merged result warns
    once per non-positive mode, not once per part file."""
    from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
    from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader

    cls = LadrunoMultiPartitionReader if ladruno else MPCOMultiPartitionReader
    paths = _two_parts(tmp_path, ladruno, lambdas=(-1e-6, -2e-6))
    with pytest.warns(UserWarning, match="non-positive eigenvalue") as rec:
        reader = cls(paths)
    with reader:
        modes = [s for s in reader.stages() if s.kind == "mode"]
    hits = [w for w in rec if "non-positive eigenvalue" in str(w.message)]
    assert len(hits) == 2                                  # one per mode
    assert {"mode 1 " in str(w.message) for w in hits} == {True, False}
    assert [(m.frequency_hz, m.period_s) for m in modes] == [(0.0, 0.0)] * 2

    # Under -W error the first (and only) warning for mode 1 raises.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="mode 1 has a non-positive"):
            cls(paths)


def test_two_eigen_solutions_get_unique_names(tmp_path: Path) -> None:
    path = tmp_path / "m.mpco"
    with h5py.File(path, "w") as f:
        f.create_group("INFO")
        _write_stage(f.create_group("MODEL_STAGE[1]"), mpco=True, rotational=False)
        data = f["MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)/DATA"]
        ds = data.create_group("STEP_5").create_dataset("MODE_0", data=_shape(0, 2))
        ds.attrs["LAMBDA"] = np.array([99.0])
    with MPCOReader(path) as r:
        modes = [s for s in r.stages() if s.kind == "mode"]
    assert [s.name for s in modes] == [
        "MODEL_STAGE[1]/STEP_2/mode_1", "MODEL_STAGE[1]/STEP_2/mode_2",
        "MODEL_STAGE[1]/STEP_5/mode_1",
    ]
    assert len({s.id for s in modes}) == 3
    assert [s.eigenvalue for s in modes] == [110.0, 4400.0, 99.0]


def test_mode_stages_take_no_program_stage_name(tmp_path: Path) -> None:
    """One static program stage and its eigen solution: the static stage
    keeps the program name, the modes keep ``mode_<k>`` (#1946)."""
    model = SimpleNamespace(stages=lambda: [SimpleNamespace(
        name="gravity", chain_attrs={"analysis": "Static"})])
    with MPCOReader(_write_mpco(tmp_path / "m.mpco")) as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error", StageCountMismatchWarning)
            _bind_stage_names(r, model, model_path="model.h5")
        stages = r.stages()
    assert [s.name for s in stages] == ["gravity", "mode_1", "mode_2"]
    assert stages[0].aliases == ("MODEL_STAGE[1]",)
    assert [s.kind for s in stages] == ["static", "mode", "mode"]


def test_ladruno_part_set_pairs_modes_beside_an_empty_part(tmp_path: Path) -> None:
    """An EMPTY_PARTITION part writes no modal group; the set still lists
    the modes, read from the part that recorded them."""
    from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader

    live = _write_ladruno(tmp_path / "r.part-0.ladruno")
    empty = tmp_path / "r.part-1.ladruno"
    with h5py.File(empty, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = max(LADRUNO_SUPPORTED_FORMAT_VERSIONS)
        stage = f.create_group("MODEL_STAGE[1]")
        stage.attrs["KIND"] = np.bytes_(b"eigen")
        stage.attrs["EMPTY_PARTITION"] = 1
    for i, p in enumerate((live, empty)):
        with h5py.File(p, "a") as f:
            f["INFO"].attrs["PARTITIONED"] = 1
            f["INFO"].attrs["PARTITION_ID"] = i
            f["INFO"].attrs["NUM_PARTITIONS"] = 2
    with LadrunoMultiPartitionReader([live, empty]) as r:
        modes = [s for s in r.stages() if s.kind == "mode"]
        assert [s.eigenvalue for s in modes] == list(_LAMBDAS)
        slab = r.read_nodes(modes[1].id, "displacement_x")
    np.testing.assert_array_equal(slab.values, _shape(1, 2)[:, 0][None, :])
