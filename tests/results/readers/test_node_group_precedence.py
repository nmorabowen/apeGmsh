"""ON_NODES groups that share a canonical name are picked by precedence (#2122).

``REACTION_FORCE`` and ``REACTION_FORCE_INCLUDING_INERTIA`` both translate
to ``reaction_force_*`` (``UNBALANCED_FORCE`` and its inertia flavor to
``force_*``, and the moments likewise). The readers used to take the first
group in HDF5 iteration order. A file written without ``track_order``
iterates by name, so ``RAYLEIGH_FORCE`` came first and a reactions read
returned the Rayleigh damping force, with no warning.

The rule now:

* the group comes from ``_NODAL_PRECEDENCE``: the static flavor, then the
  inertia-including one, in any creation or iteration order;
* a Rayleigh group is the damping force alone (OpenSees
  ``calculateNodalReactions(2)``), never a reaction: it translates to no
  canonical name, and a read of ``reaction_*`` from a stage that records
  only Rayleigh groups is refused by name;
* when the leading group lacks the requested column but a lower one holds
  it, the read is refused rather than mixing groups across axes.

Each group holds distinct values (``REACTION_FORCE`` x1, the inertia
flavor x100, Rayleigh x10000), so the oracle names the group read. The
synthetic files are written with ``h5py``; the last test is the live
oracle against ``ops.nodeReaction`` on the fork.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import _mpco_translation as mtr
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
from apeGmsh.results.readers._mpco_translation import NodeComponentAmbiguous
from apeGmsh.results.readers._protocol import ResultLevel

_TIME = np.array([0.5, 1.0])

#: Scale per group, so a value names the group it came from.
_SCALE = {
    "REACTION_FORCE": 1.0,
    "REACTION_FORCE_INCLUDING_INERTIA": 100.0,
    "RAYLEIGH_FORCE": 10000.0,
    "REACTION_FORCE_RAYLEIGH": 10000.0,
    "REACTION_MOMENT": 2.0,
    "REACTION_MOMENT_INCLUDING_INERTIA": 200.0,
    "RAYLEIGH_MOMENT": 20000.0,
    "UNBALANCED_FORCE": 3.0,
    "UNBALANCED_FORCE_INCLUDING_INERTIA": 300.0,
    "UNBALANCED_MOMENT": 4.0,
    "UNBALANCED_MOMENT_INCLUDING_INERTIA": 400.0,
    "DISPLACEMENT": 0.001,
}
_LABELS = {"FORCE": "Px,Py", "MOMENT": "Mz"}

_REACTIONS = (
    "RAYLEIGH_FORCE", "REACTION_FORCE", "REACTION_FORCE_INCLUDING_INERTIA",
)
_ORDERS = {
    "alphabetical": tuple(sorted(_REACTIONS)),
    "reverse": tuple(sorted(_REACTIONS, reverse=True)),
    # The fork recorder keeps the -N token order; this one puts Rayleigh
    # first, then the inertia flavor.
    "fork_tokens": (
        "RAYLEIGH_FORCE", "REACTION_FORCE_INCLUDING_INERTIA", "REACTION_FORCE",
    ),
}


def _labels(name: str) -> str:
    return _LABELS["MOMENT" if "MOMENT" in name else "FORCE"]


def _values(name: str, nodes: Iterable[int], labels: str) -> np.ndarray:
    """``(T, n, nComp)``: ``scale * (10 * node + col + 1)``, half at step 0."""
    ncomp = len(labels.split(","))
    last = np.array([
        [_SCALE[name] * (10 * n + c + 1) for c in range(ncomp)] for n in nodes
    ])
    return np.stack([0.5 * last, last])


def _expected(name: str, nodes: Iterable[int], col: int) -> np.ndarray:
    """``(T, n)`` of column ``col`` of group ``name``."""
    return _values(name, nodes, _labels(name))[:, :, col]


# ---------------------------------------------------------------------
# .ladruno / .mpco writers
# ---------------------------------------------------------------------

def _ladruno(
    path: Path, groups: Iterable[str], *, track_order: bool = False,
    nodes: tuple[int, ...] = (1, 2), labels: "dict[str, str] | None" = None,
    part: "int | None" = None,
) -> Path:
    labels = labels or {}
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = 1
        if part is not None:
            info.attrs["PARTITIONED"] = 1
            info.attrs["PARTITION_ID"] = part
            info.attrs["NUM_PARTITIONS"] = 2
        stage = f.create_group("MODEL_STAGE[1]")
        stage.attrs["KIND"] = np.bytes_(b"static")
        res = stage.create_group("RESULTS")
        on_nodes = res.create_group("ON_NODES", track_order=track_order)
        for name in groups:
            comps = labels.get(name, _labels(name))
            grp = on_nodes.create_group(name)
            grp.attrs["COMPONENTS"] = np.array([comps.encode()])
            grp.create_dataset(
                "ID", data=np.array(nodes, dtype=np.int64).reshape(-1, 1),
            )
            grp.create_dataset("DATA", data=_values(name, nodes, comps))
            grp.create_dataset("TIME", data=_TIME)
            grp.create_dataset("STEP", data=np.arange(_TIME.size))
    return path


def _mpco(
    path: Path, groups: Iterable[str], *, track_order: bool = False,
    nodes: tuple[int, ...] = (1, 2), labels: "dict[str, str] | None" = None,
) -> Path:
    labels = labels or {}
    ids = np.array(nodes, dtype=np.int32).reshape(-1, 1)
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.create_dataset("SPATIAL_DIM", data=2)
        stage = f.create_group("MODEL_STAGE[1]")
        stage.attrs["STEP"] = 0
        stage.attrs["TIME"] = 0.0
        model = stage.create_group("MODEL")
        model.create_group("NODES").create_dataset("ID", data=ids)
        model["NODES"].create_dataset(
            "COORDINATES", data=np.zeros((ids.size, 2)),
        )
        model.create_group("ELEMENTS")
        res = stage.create_group("RESULTS")
        on_nodes = res.create_group("ON_NODES", track_order=track_order)
        for name in groups:
            comps = labels.get(name, _labels(name))
            data = _values(name, nodes, comps)
            grp = on_nodes.create_group(name)
            grp.attrs["COMPONENTS"] = np.array([comps.encode()])
            grp.create_dataset("ID", data=ids)
            steps = grp.create_group("DATA")
            for k in range(_TIME.size):
                ds = steps.create_dataset(f"STEP_{k}", data=data[k])
                ds.attrs["STEP"] = k
                ds.attrs["TIME"] = _TIME[k]
    return path


_FORMATS = {"ladruno": (_ladruno, LadrunoReader), "mpco": (_mpco, MPCOReader)}


def _read(fmt: str, path: Path, component: str) -> np.ndarray:
    reader_cls = _FORMATS[fmt][1]
    with reader_cls(path) as r:
        slab = r.read_nodes(r.stages()[0].id, component)
    assert slab.node_ids.tolist() == [1, 2]
    return slab.values


def _write(fmt: str, tmp_path: Path, groups, **kw) -> Path:
    return _FORMATS[fmt][0](tmp_path / f"run.{fmt}", groups, **kw)


# ---------------------------------------------------------------------
# Precedence, in every creation / iteration order
# ---------------------------------------------------------------------

@pytest.mark.parametrize("fmt", sorted(_FORMATS))
@pytest.mark.parametrize("track_order", [False, True])
@pytest.mark.parametrize("order", sorted(_ORDERS))
def test_reaction_force_reads_reaction_force(
    tmp_path: Path, fmt: str, order: str, track_order: bool,
) -> None:
    path = _write(fmt, tmp_path, _ORDERS[order], track_order=track_order)
    for col, axis in enumerate("xy"):
        np.testing.assert_array_equal(
            _read(fmt, path, f"reaction_force_{axis}"),
            _expected("REACTION_FORCE", (1, 2), col),
        )


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
@pytest.mark.parametrize("track_order", [False, True])
def test_inertia_flavor_reads_when_static_is_absent(
    tmp_path: Path, fmt: str, track_order: bool,
) -> None:
    path = _write(
        fmt, tmp_path,
        ("RAYLEIGH_FORCE", "REACTION_FORCE_INCLUDING_INERTIA"),
        track_order=track_order,
    )
    np.testing.assert_array_equal(
        _read(fmt, path, "reaction_force_y"),
        _expected("REACTION_FORCE_INCLUDING_INERTIA", (1, 2), 1),
    )


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
@pytest.mark.parametrize(
    "groups, component", [
        (("RAYLEIGH_FORCE",), "reaction_force_x"),
        (("RAYLEIGH_MOMENT",), "reaction_moment_z"),
        (("REACTION_FORCE_RAYLEIGH",), "reaction_force_x"),
    ],
)
def test_rayleigh_alone_is_not_a_reaction(
    tmp_path: Path, fmt: str, groups, component: str,
) -> None:
    """Rayleigh forces never satisfy ``reaction_*``: refused by name."""
    path = _write(fmt, tmp_path, groups + ("DISPLACEMENT",))
    reader_cls = _FORMATS[fmt][1]
    with reader_cls(path) as r:
        stage = r.stages()[0].id
        listed = r.available_components(stage, ResultLevel.NODES)
        assert not [c for c in listed if c.startswith("reaction")]
        with pytest.raises(NodeComponentAmbiguous, match=groups[0]):
            r.read_nodes(stage, component)
        # The other groups of the stage still read.
        assert r.read_nodes(stage, "displacement_x").values.size


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
@pytest.mark.parametrize("track_order", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_other_many_to_one_canonicals(
    tmp_path: Path, fmt: str, track_order: bool, reverse: bool,
) -> None:
    """The moments and the unbalanced loads follow the same precedence."""
    groups = sorted(
        ["REACTION_MOMENT", "REACTION_MOMENT_INCLUDING_INERTIA",
         "RAYLEIGH_MOMENT", "UNBALANCED_FORCE",
         "UNBALANCED_FORCE_INCLUDING_INERTIA", "UNBALANCED_MOMENT",
         "UNBALANCED_MOMENT_INCLUDING_INERTIA"],
        reverse=reverse,
    )
    path = _write(fmt, tmp_path, groups, track_order=track_order)
    for component, group, col in [
        ("reaction_moment_z", "REACTION_MOMENT", 0),
        ("force_x", "UNBALANCED_FORCE", 0),
        ("force_y", "UNBALANCED_FORCE", 1),
        ("moment_z", "UNBALANCED_MOMENT", 0),
    ]:
        np.testing.assert_array_equal(
            _read(fmt, path, component), _expected(group, (1, 2), col),
        )


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_leading_group_without_the_column_is_refused(
    tmp_path: Path, fmt: str,
) -> None:
    """Never ``reaction_force_x`` from one group and ``_z`` from another."""
    path = _write(
        fmt, tmp_path, ("REACTION_FORCE", "REACTION_FORCE_INCLUDING_INERTIA"),
        labels={"REACTION_FORCE_INCLUDING_INERTIA": "Px,Py,Pz"},
    )
    reader_cls = _FORMATS[fmt][1]
    with reader_cls(path) as r:
        stage = r.stages()[0].id
        with pytest.raises(NodeComponentAmbiguous, match="REACTION_FORCE"):
            r.read_nodes(stage, "reaction_force_z")
        np.testing.assert_array_equal(
            r.read_nodes(stage, "reaction_force_x").values,
            _expected("REACTION_FORCE", (1, 2), 0),
        )


# ---------------------------------------------------------------------
# Partitioned readers give the same answer
# ---------------------------------------------------------------------

@pytest.mark.parametrize("track_order", [False, True])
def test_partitioned_ladruno_sums_reaction_force(
    tmp_path: Path, track_order: bool,
) -> None:
    order = _ORDERS["fork_tokens" if track_order else "alphabetical"]
    paths = [
        _ladruno(tmp_path / "run.part-0.ladruno", order, nodes=(1, 2),
                 track_order=track_order, part=0),
        _ladruno(tmp_path / "run.part-1.ladruno", order, nodes=(1, 3),
                 track_order=track_order, part=1),
    ]
    _assert_shared_sum(LadrunoMultiPartitionReader(paths))


@pytest.mark.parametrize("track_order", [False, True])
def test_partitioned_mpco_sums_reaction_force(
    tmp_path: Path, track_order: bool,
) -> None:
    order = _ORDERS["fork_tokens" if track_order else "alphabetical"]
    paths = [
        _mpco(tmp_path / "run.part-0.mpco", order, nodes=(1, 2),
              track_order=track_order),
        _mpco(tmp_path / "run.part-1.mpco", order, nodes=(1, 3),
              track_order=track_order),
    ]
    _assert_shared_sum(MPCOMultiPartitionReader(paths))


def _assert_shared_sum(reader) -> None:
    try:
        slab = reader.read_nodes("stage_0", "reaction_force_y")
        got = {int(n): slab.values[:, i] for i, n in enumerate(slab.node_ids)}
    finally:
        reader.close()
    one = _expected("REACTION_FORCE", (1,), 1)[:, 0]
    # Node 1 is on both partitions: the two REACTION_FORCE partials sum.
    np.testing.assert_array_equal(got[1], 2 * one)
    np.testing.assert_array_equal(got[2], _expected("REACTION_FORCE", (2,), 1)[:, 0])
    np.testing.assert_array_equal(got[3], _expected("REACTION_FORCE", (3,), 1)[:, 0])


# ---------------------------------------------------------------------
# The translation table itself
# ---------------------------------------------------------------------

def test_every_many_to_one_prefix_has_a_total_precedence(monkeypatch) -> None:
    mtr._check_precedence()
    monkeypatch.setitem(
        mtr._NODAL_PRECEDENCE, "reaction_force", ("REACTION_FORCE",),
    )
    with pytest.raises(RuntimeError, match="reaction_force"):
        mtr._check_precedence()


def test_rayleigh_groups_translate_to_no_canonical() -> None:
    for name in ("RAYLEIGH_FORCE", "RAYLEIGH_MOMENT",
                 "REACTION_FORCE_RAYLEIGH", "REACTION_MOMENT_RAYLEIGH"):
        assert mtr.canonical_node_component(name, "Px") is None
        assert not mtr.has_canonical_mapping(name)


def test_reverse_lookup_follows_precedence() -> None:
    assert mtr.canonical_to_mpco_lookup("reaction_force_x") == (
        "REACTION_FORCE", "x",
    )
    assert mtr.canonical_to_mpco_lookup("moment_z") == (
        "UNBALANCED_MOMENT", "z",
    )


# ---------------------------------------------------------------------
# Live oracle: the fork's recorders against ops.nodeReaction
# ---------------------------------------------------------------------

@pytest.mark.ladruno_fork
def test_live_reactions_match_node_reaction(tmp_path: Path) -> None:
    """A damped transient portal frame records ``rayleighForce``, then
    ``reactionForceIncludingInertia``, then ``reactionForce`` (the
    ``.ladruno`` recorder keeps that order; Rayleigh iterates first). Both
    readers return ``ops.nodeReaction`` after ``ops.reactions()`` at
    ``rtol=1e-10``, at every step and support DOF."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    paths = {"ladruno": tmp_path / "run.ladruno", "mpco": tmp_path / "run.mpco"}
    supports = (1, 4)
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, x, y in ((1, 0.0, 0.0), (2, 0.0, 3.0), (3, 4.0, 3.0),
                          (4, 4.0, 0.0)):
            ops.node(tag, x, y)
        ops.fix(1, 1, 1, 1)
        ops.fix(4, 1, 1, 0)
        for tag in (2, 3):
            ops.mass(tag, 1.0e3, 1.0e3, 0.0)
        ops.geomTransf("Linear", 1)
        for tag, (i, j) in enumerate(((1, 2), (2, 3), (3, 4)), start=1):
            ops.element("elasticBeamColumn", tag, i, j, 0.01, 2.0e11, 1.0e-4, 1)
        ops.rayleigh(0.3, 0.0, 0.0, 2.0e-3)
        for kind, path in paths.items():
            ops.recorder(kind, str(path), "-N", "rayleighForce",
                         "reactionForceIncludingInertia", "reactionForce")
        ops.timeSeries("Sine", 1, 0.0, 10.0, 0.5)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 1.0e4, 0.0, 0.0)
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-12, 10)
        ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")
        ref = []
        for _ in range(5):
            assert ops.analyze(1, 0.01) == 0
            ops.reactions()
            ref.append([[ops.nodeReaction(n, d) for n in supports]
                        for d in (1, 2)])
    finally:
        ops.wipe()  # closes the recorders
    want = np.asarray(ref)  # (T, dof, node)
    assert np.abs(want).max() > 1.0

    with h5py.File(paths["ladruno"], "r") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE"))
        names = list(f[stage]["RESULTS/ON_NODES"])
    # The precondition the test is about: Rayleigh iterates first.
    assert names[0] == "RAYLEIGH_FORCE", names

    for fmt, path in paths.items():
        reader_cls = _FORMATS[fmt][1]
        with reader_cls(path) as r:
            stage = r.stages()[0].id
            for d, axis in enumerate("xy"):
                slab = r.read_nodes(
                    stage, f"reaction_force_{axis}",
                    node_ids=np.array(supports),
                )
                assert slab.node_ids.tolist() == list(supports), fmt
                np.testing.assert_allclose(
                    slab.values, want[:, d, :], rtol=1e-10, atol=0.0,
                    err_msg=f"{fmt} reaction_force_{axis}",
                )
