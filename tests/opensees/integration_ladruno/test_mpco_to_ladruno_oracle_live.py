"""#2120 — ADR 0122 D8: the MPCO -> ``.ladruno`` conformance oracle.

One bridge model (a 2-D portal: a fiber ``forceBeamColumn`` column and an
``elasticBeamColumn`` beam, lumped masses) runs one static stage
(``gravity``) and then ``eigen``; the deck records an ``.mpco`` and a
``.ladruno`` in the same run, on the fork, in a fresh process.
``convert_mpco(.mpco)`` must then:

1. pass the fork's own ``ladruno_format.validate()`` (fetched at
   ``FORK_PIN``), returning ``[]``;
2. equal the fork's ``.ladruno`` under :func:`compare`, an
   information-equality comparator: the fork's ``normalize_nodal`` and
   ``normalize_element`` (values keyed by id, gauss point, section, fiber,
   component and step), the modal groups, ``NODES``, ``CONNECTIVITY``,
   ``SECTION_ASSIGNMENTS``, the result and bucket names (in order), the
   ``STEP`` / ``TIME`` axes and the BASIS of every group the converter
   describes; what it leaves out must equal the declared
   ``PROVENANCE@unsupported``;
3. fail :func:`compare` after either mutation, a swapped column or a
   dropped bucket, which ``validate()`` alone passes (it checks shapes).

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import convert_mpco

from tests.opensees.integration_ladruno import _fork_spec

pytestmark = pytest.mark.ladruno_fork

E, A, I_BEAM, MASS, P = 2.0e11, 0.02, 2.0e-4, 50.0, 1.0e4
N_MODES = 3
NODAL = ("displacement", "reactionForce", "modesOfVibration",
         "modesOfVibrationRotational")
ELEMENTS = ("force", "localForce", "basicForce", "section.force",
            "section.fiber.stress")
_STAGE = "MODEL_STAGE["
# Values from one run are the same doubles in both files; the tolerance is
# the fork's own element parity check's (element_parity_check.py).
_ATOL, _RTOL = 1e-12, 1e-9
# The attributes both recorders write on a result (MPCO's own set; the fork
# may add PARTITION_REDUCTION, which an .mpco never carries).
_NODE_ATTRS = ("COMPONENTS", "DATA_TYPE", "DESCRIPTION", "DIMENSION",
               "DISPLAY_NAME", "TYPE")
_BUCKET_ATTRS = ("DATA_TYPE", "DISPLAY_NAME", "NUM_COLUMNS", "TYPE")


def _portal_fem() -> Any:
    import gmsh

    from apeGmsh import apeGmsh as Session

    with Session(model_name="mpco_to_ladruno_portal", verbose=False) as g:
        geo = g.model.geometry
        base = geo.add_point(0.0, 0.0, 0.0)
        top = geo.add_point(0.0, 3.0, 0.0)
        corner = geo.add_point(4.0, 3.0, 0.0)
        col = geo.add_line(base, top)
        beam = geo.add_line(top, corner)
        for line in (col, beam):
            gmsh.model.mesh.setTransfiniteCurve(line, 3)
        g.physical.add(1, [col], name="Col")
        g.physical.add(1, [beam], name="Beam")
        g.labels.add(0, [base], "base")
        g.labels.add(0, [top], "top")
        g.labels.add(0, [corner], "corner")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _fresh_env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    return env


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    tmp = tmp_path_factory.mktemp("mpco_to_ladruno_oracle")
    mpco, fork = tmp / "r.mpco", tmp / "r.ladruno"
    model_h5, deck = tmp / "model.h5", tmp / "deck.py"
    fem = _portal_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    steel = ops.uniaxialMaterial.ElasticMaterial(E=E)
    sec = ops.section.Fiber(fibers=tuple(
        FiberPoint(material=steel, y=y, z=0.0, area=A / 4)
        for y in (-0.15, -0.05, 0.05, 0.15)))
    ops.element.forceBeamColumn(
        pg="Col", transf=t,
        integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3))
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=A, E=E, Iz=I_BEAM)
    ops.fix(label="base", dofs=(1, 1, 1))
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    free = ids[np.linalg.norm(xyz, axis=1) > 0.0]  # every node but the base
    ops.mass(nodes=[int(n) for n in free], values=(MASS, MASS, 0.0))
    ops.recorder.MPCO(file=str(mpco), nodal_responses=NODAL,
                      elem_responses=ELEMENTS)
    ops.recorder.Ladruno(file=str(fork), nodal_responses=NODAL,
                         elem_responses=ELEMENTS)
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=ops.nodes.one(label="top"), forces=(P, -P, 0.0))
            p.load(node=ops.nodes.one(label="corner"), forces=(0.0, -P, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
            system=ops.system.BandGeneral(), analysis=ops.analysis.Static(),
        )
        s.run(n_increments=2)
    ops.h5(str(model_h5))
    ops.py(str(deck), progress=False)

    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n",
    )
    # The eigen solution after the stage, recorded once; then close the
    # recorders and prove the deck ran to its end.
    text += (f"\nops.eigen({N_MODES})\nops.record()\n"
             "ops.remove('recorders')\nprint('DECK-DONE')\n")
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0 and "DECK-DONE" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-2000:])
    converted = convert_mpco(mpco, tmp / "converted.ladruno",
                             model_h5=model_h5)
    assert converted == [tmp / "converted.ladruno"]
    return {"tmp": tmp, "mpco": mpco, "fork": fork, "model_h5": model_h5,
            "converted": converted[0]}


@pytest.fixture(scope="module")
def lf(run: "dict[str, Path]") -> Iterator[ModuleType]:
    spec = _fork_spec.fetch(run["tmp"] / "fork_spec")
    with _fork_spec.ladruno_format(spec) as module:
        yield module


# ---------------------------------------------------------------------------
# The information-equality comparator
# ---------------------------------------------------------------------------


def _stages(f: "h5py.File") -> list[str]:
    return sorted((k for k in f if k.startswith(_STAGE)),
                  key=lambda n: int(n[len(_STAGE):-1]))


def _modal(name: str) -> bool:
    return name.startswith("MODES_OF_VIBRATION")


class _OnNodesView:
    """``reader.f`` with the modal groups hidden from one ``ON_NODES``.

    The fork's ``normalize_nodal`` iterates every ``ON_NODES`` group as a
    step series; a modal group (``DATA/STEP_k/MODE_i``) is compared on its
    own instead.
    """

    def __init__(self, f: "h5py.File", path: str) -> None:
        self._f, self._path = f, path

    def __contains__(self, path: str) -> bool:
        return path in self._f

    def __getitem__(self, path: str) -> Any:
        grp = self._f[path]
        if path != self._path:
            return grp
        return _Filtered(grp, [n for n in grp if not _modal(n)])


class _Filtered:
    def __init__(self, grp: "h5py.Group", names: "list[str]") -> None:
        self._grp, self._names = grp, names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __getitem__(self, name: str) -> Any:
        return self._grp[name]


def _close(a: "dict[Any, float]", b: "dict[Any, float]", what: str) -> list[str]:
    out = []
    if a.keys() != b.keys():
        only_a = sorted(set(a) - set(b), key=str)[:3]
        only_b = sorted(set(b) - set(a), key=str)[:3]
        out.append(f"{what}: keys differ (fork only {only_a}, converted "
                   f"only {only_b})")
    for k in a.keys() & b.keys():
        if not abs(a[k] - b[k]) <= _ATOL + _RTOL * abs(a[k]):
            out.append(f"{what}: {k} fork={a[k]!r} converted={b[k]!r}")
            break
    return out


def _arrays(a: Any, b: Any, what: str) -> list[str]:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or not np.array_equal(x, y):
        return [f"{what}: {x.shape} vs {y.shape} or values differ"]
    return []


def _attr(obj: Any, name: str) -> Any:
    if name not in obj.attrs:
        return None
    v = np.asarray(obj.attrs[name]).ravel()
    v = v[0] if v.size == 1 else v
    return v.decode() if isinstance(v, bytes) else v


def compare(
    fork: Path, converted: Path, lf: ModuleType,
) -> "tuple[list[str], set[str]]":
    """``(mismatches, lossy)`` between a fork ``.ladruno`` and a conversion.

    ``lossy`` names, in ``PROVENANCE@unsupported`` vocabulary, what the
    fork file holds and the converted file lacks (``local_axes``,
    ``quadrature:<group>``); everything else must be equal.
    """
    problems: list[str] = []
    lossy: set[str] = set()
    with lf.LadrunoReader(str(fork)) as ra, lf.LadrunoReader(str(converted)) as rb:
        fa, fb = ra.f, rb.f
        sa, sb = _stages(fa), _stages(fb)
        if len(sa) != len(sb):
            return [f"stage count: fork {len(sa)}, converted {len(sb)}"], lossy
        for ga, gb in zip(sa, sb):
            where = f"{ga}|{gb}"
            ida, xa = ra.nodes(ga)
            idb, xb = rb.nodes(gb)
            problems += _arrays(ida, idb, f"{where} NODES/ID")
            problems += _arrays(xa, xb, f"{where} NODES/COORDINATES")
            ea, eb = ra.element_groups(ga), rb.element_groups(gb)
            if list(ea) != list(eb):
                problems.append(f"{where} ELEMENTS: {list(ea)} vs {list(eb)}")
            for name in ea.keys() & eb.keys():
                problems += _arrays(ea[name]["connectivity"],
                                    eb[name]["connectivity"],
                                    f"{where} {name} CONNECTIVITY")
                if eb[name]["gp_param"] is None:
                    if ea[name]["gp_param"] is not None:
                        lossy.add(f"quadrature:{name}")
                    continue
                for key in ("topology", "family", "param_domain", "num_gp",
                            "ndir", "num_ctrl", "rational"):
                    if ea[name][key] != eb[name][key]:
                        problems.append(
                            f"{where} {name} {key}: {ea[name][key]!r} vs "
                            f"{eb[name][key]!r}")
                problems += _arrays(ea[name]["gp_param"], eb[name]["gp_param"],
                                    f"{where} {name} GP_PARAM")
            if ra.local_axes(ga) and not rb.local_axes(gb):
                lossy.add("local_axes")
            sec_a, sec_b = ra.section_assignments(ga), rb.section_assignments(gb)
            if list(sec_a) != list(sec_b):
                problems.append(f"{where} SECTIONS: {list(sec_a)} vs {list(sec_b)}")
            for name in sec_a.keys() & sec_b.keys():
                for key in ("kind", "assignment", "fiber_data", "fiber_materials"):
                    problems += _arrays(sec_a[name].get(key), sec_b[name].get(key),
                                        f"{where} {name} {key}")
            problems += _results(fa, fb, ga, gb, lf, where)
    return problems, lossy


def _results(
    fa: "h5py.File", fb: "h5py.File", ga: str, gb: str, lf: ModuleType,
    where: str,
) -> list[str]:
    problems: list[str] = []
    na, nb = f"{ga}/RESULTS/ON_NODES", f"{gb}/RESULTS/ON_NODES"
    names_a = list(fa[na]) if na in fa else []
    names_b = list(fb[nb]) if nb in fb else []
    if names_a != names_b:
        problems.append(f"{where} ON_NODES: {names_a} vs {names_b}")
    ea, eb = f"{ga}/RESULTS/ON_ELEMENTS", f"{gb}/RESULTS/ON_ELEMENTS"
    buckets_a = [(t, b) for t in (fa[ea] if ea in fa else ()) for b in fa[ea][t]]
    buckets_b = [(t, b) for t in (fb[eb] if eb in fb else ()) for b in fb[eb][t]]
    if buckets_a != buckets_b:
        problems.append(f"{where} buckets: {sorted(set(buckets_a) ^ set(buckets_b))}")
    # Step series: the fork's normalizers, keyed by information.
    for normalize, label in ((lf.normalize_nodal, "nodal"),
                             (lf.normalize_element, "element")):
        va = normalize(SimpleNamespace(f=_OnNodesView(fa, na)), ga)
        vb = normalize(SimpleNamespace(f=_OnNodesView(fb, nb)), gb)
        if not va:
            problems.append(f"{where} {label}: the fork file normalizes to "
                            "nothing; the oracle would compare no data")
        problems += _close(va, vb, f"{where} {label}")
    # Self-description: result attributes, and each bucket's COLUMN_MAP.
    for name in (n for n in names_a if n in names_b):
        for key in _NODE_ATTRS:
            if _attr(fa[f"{na}/{name}"], key) != _attr(fb[f"{nb}/{name}"], key):
                problems.append(f"{where} ON_NODES/{name} @{key}")
    for token, bucket in (k for k in buckets_a if k in set(buckets_b)):
        pa, pb = f"{ea}/{token}/{bucket}", f"{eb}/{token}/{bucket}"
        for key in _BUCKET_ATTRS:
            if _attr(fa[pa], key) != _attr(fb[pb], key):
                problems.append(f"{where} {token}/{bucket} @{key}")
        ca, cb = fa[f"{pa}/COLUMN_MAP"], fb[f"{pb}/COLUMN_MAP"]
        if sorted(ca) != sorted(cb):
            problems.append(f"{where} {token}/{bucket} COLUMN_MAP children")
            continue
        for key in ca:
            problems += _arrays(ca[key], cb[key],
                                f"{where} {token}/{bucket} COLUMN_MAP/{key}")
        if _attr(ca, "COMP_NAMES") != _attr(cb, "COMP_NAMES"):
            problems.append(f"{where} {token}/{bucket} @COMP_NAMES")
    # STEP / TIME axes of every step series.
    for base_a, base_b, items in (
            (na, nb, [(n,) for n in names_a if not _modal(n)]),
            (ea, eb, buckets_a)):
        for item in items:
            pa, pb = "/".join((base_a, *item)), "/".join((base_b, *item))
            if pb not in fb:
                continue
            for axis in ("STEP", "TIME"):
                problems += _arrays(fa[pa][axis], fb[pb][axis],
                                    f"{where} {'/'.join(item)} {axis}")
    # Modes: every MODE_i dataset and its eigen attributes.
    for name in (n for n in names_a if _modal(n) and n in names_b):
        da, db = fa[f"{na}/{name}/DATA"], fb[f"{nb}/{name}/DATA"]
        problems += _arrays(fa[f"{na}/{name}/ID"], fb[f"{nb}/{name}/ID"],
                            f"{where} {name} ID")
        if sorted(da) != sorted(db):
            problems.append(f"{where} {name}: steps {sorted(da)} vs {sorted(db)}")
            continue
        for step in da:
            if sorted(da[step]) != sorted(db[step]):
                problems.append(f"{where} {name}/{step}: modes differ")
                continue
            for mode in da[step]:
                problems += _arrays(da[step][mode], db[step][mode],
                                    f"{where} {name}/{step}/{mode}")
                for key in ("LAMBDA", "OMEGA", "FREQUENCY", "PERIOD", "MODE"):
                    if _attr(da[step][mode], key) != _attr(db[step][mode], key):
                        problems.append(f"{where} {name}/{step}/{mode} @{key}")
    return problems


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


def test_the_fork_and_mpco_stage_counts_agree(run) -> None:
    """Card stop condition: the two recorders split the run alike."""
    with h5py.File(run["fork"], "r") as a, h5py.File(run["mpco"], "r") as b:
        assert len(_stages(a)) == len(_stages(b)) == 1


def test_converted_file_passes_the_fork_validator(run, lf) -> None:
    assert lf.validate(str(run["converted"])) == []
    assert lf.validate(str(run["fork"])) == []  # the reference conforms too


def test_converted_file_equals_the_fork_file(run, lf) -> None:
    problems, lossy = compare(run["fork"], run["converted"], lf)
    assert problems == []
    with h5py.File(run["converted"], "r") as f:
        unsupported = json.loads(f["INFO/PROVENANCE"].attrs["unsupported"])
        assert f["INFO/PROVENANCE"].attrs["writer"].startswith(
            "apeGmsh mpco->ladruno ")
    # What the comparator had to leave out is exactly what was declared.
    assert set(unsupported) == lossy
    assert "quadrature:3-ElasticBeam2d[1:0]" in lossy  # the oracle bites


def _mutant(run: "dict[str, Path]", name: str) -> Path:
    out = run["tmp"] / f"mutant_{name}.ladruno"
    shutil.copy(run["converted"], out)
    return out


_BUCKET = "RESULTS/ON_ELEMENTS/localForce/3-ElasticBeam2d[1:0:0]"


def test_a_swapped_column_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "swap")
    with h5py.File(path, "r+") as f:
        data = f[f"{_stages(f)[0]}/{_BUCKET}/DATA"]
        values = data[...]
        # The first two columns that hold different values (N_1 and V_1 of
        # the loaded beam), so the swap changes the information.
        i, j = next((i, j) for i in range(values.shape[2])
                    for j in range(i + 1, values.shape[2])
                    if not np.allclose(values[..., i], values[..., j]))
        values[..., [i, j]] = values[..., [j, i]]
        data[...] = values
    assert lf.validate(str(path)) == []  # shapes only: validate() passes it
    problems, _ = compare(run["fork"], path, lf)
    assert any("element" in p for p in problems), problems


def test_a_dropped_bucket_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "drop")
    with h5py.File(path, "r+") as f:
        del f[f"{_stages(f)[0]}/{_BUCKET}"]
    assert lf.validate(str(path)) == []  # shapes only: validate() passes it
    problems, _ = compare(run["fork"], path, lf)
    assert any("buckets" in p for p in problems), problems
    assert any("keys differ" in p for p in problems), problems
