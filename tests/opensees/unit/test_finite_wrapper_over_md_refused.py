"""#2502: a finite-strain lift over a SANISAND inner is refused at declaration.

The fork's ``LogStrain`` / ``LogStrain2D`` ask their inner for
``getType()`` at the wrapper's ``nDMaterial`` line. ``ManzariDafalias``,
``LadrunoSANISAND`` and ``SAniSandMS`` do not implement it, so the fork
exits the process there (rc 255, ``<Class>::getType -- subclass
responsibility``; #2499). The bridge refuses exactly those pairs, by name,
through :data:`~apeGmsh.opensees.material.nd.FINITE_LIFT_ABORTING_INNERS`.

Two layers:

* unit tests (no backend): each refused pair raises at construction and
  through the namespace; the pairs that do not abort still build;
* a live-fork probe (``ladruno_fork``) that runs every (inner x wrapper
  chain) pair in a child interpreter on the pinned build and pins what
  each did. It is the oracle for the refused set: when the fork fixes
  #2499 the probe fails first and says to empty the set.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.material import nd
from apeGmsh.opensees.material.nd import (
    FINITE_LIFT_ABORTING_INNERS,
    ElasticIsotropic,
    InitDefGrad,
    LadrunoSANISAND,
    LogStrain,
    LogStrain2D,
    ManzariDafalias,
    NDMaterial,
    SAniSandMS,
    StagedStrain,
)

_ROOT = Path(__file__).resolve().parents[3]

_GORINI: dict[str, float] = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "rho": 2.0,
}
_MD_KW: dict[str, float] = {**_GORINI, "z_max": 12.5, "cz": 1100.0}
_MS_KW: dict[str, float] = {**_GORINI, "zeta": 0.0005, "mu0": 0.26,
                            "beta": 1.0}


def _leaf(name: str) -> NDMaterial:
    if name == "ManzariDafalias":
        return ManzariDafalias(**_MD_KW)
    if name == "LadrunoSANISAND":
        return LadrunoSANISAND(**_MD_KW)
    if name == "SAniSandMS":
        return SAniSandMS(**_MS_KW)
    if name == "ElasticIsotropic":
        return ElasticIsotropic(E=2.0e5, nu=0.25)
    raise AssertionError(f"unknown leaf {name!r}")


_SANDS = ("ManzariDafalias", "LadrunoSANISAND", "SAniSandMS")
#: The fork class that prints the exit message, per inner (measured:
#: LadrunoSANISAND exits through its ManzariDafalias base).
_EXITS_AS = {"ManzariDafalias": "ManzariDafalias",
             "LadrunoSANISAND": "ManzariDafalias",
             "SAniSandMS": "SAniSandMS"}
_LIFTS = (LogStrain, LogStrain2D)


# ---------------------------------------------------------------------------
# Unit layer
# ---------------------------------------------------------------------------


def test_the_gate_names_exactly_the_measured_sands() -> None:
    """The set is what the live probe measured to abort; every name is a
    real nd class, so a typo cannot silently disarm the gate."""
    assert FINITE_LIFT_ABORTING_INNERS == _EXITS_AS
    for name in FINITE_LIFT_ABORTING_INNERS:
        cls = getattr(nd, name)
        assert isinstance(cls, type) and issubclass(cls, NDMaterial), name


@pytest.mark.parametrize("sand", _SANDS)
@pytest.mark.parametrize("lift", _LIFTS, ids=lambda c: c.__name__)
def test_a_lift_over_a_sand_is_refused_by_name(lift: type, sand: str) -> None:
    with pytest.raises(ValueError) as exc:
        lift(inner=_leaf(sand))
    msg = str(exc.value)
    assert msg.startswith(f"{lift.__name__}: inner {sand} ")
    assert "#2499" in msg
    assert f"{_EXITS_AS[sand]}::getType -- subclass responsibility" in msg


def _bridge() -> apeSees:
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


@pytest.mark.parametrize("sand", ["ManzariDafalias", "LadrunoSANISAND"])
@pytest.mark.parametrize("lift", ["LogStrain", "LogStrain2D"])
def test_the_namespace_refuses_by_registered_name(lift: str, sand: str) -> None:
    """The refusal sits in the dataclass, so the namespace route (inner
    passed as its registered name) cannot reach the fork either."""
    ops = _bridge()
    kw = _MD_KW
    getattr(ops.nDMaterial, sand)(name="sand", **kw)
    with pytest.raises(ValueError, match=r"#2499"):
        getattr(ops.nDMaterial, lift)(inner="sand")


def test_init_def_grad_over_a_lifted_sand_is_refused_at_the_lift() -> None:
    """``InitDefGrad(LogStrain(MD))`` aborts at its inner ``LogStrain``
    line; the inner's refusal is what stops it."""
    with pytest.raises(ValueError, match=r"LogStrain: inner ManzariDafalias"):
        InitDefGrad(inner=LogStrain(inner=_leaf("ManzariDafalias")))


@pytest.mark.parametrize("lift", _LIFTS, ids=lambda c: c.__name__)
def test_an_isotropic_inner_still_builds_and_emits(lift: type) -> None:
    inner = _leaf("ElasticIsotropic")
    mat = lift(inner=inner)
    rec = RecordingEmitter()
    set_tag_resolver(rec, lambda p: 7 if p is inner else 0)
    mat._emit(rec, tag=12)
    assert rec.calls == [("nDMaterial", (lift.__name__, 12, 7), {})]


@pytest.mark.parametrize("sand", _SANDS)
@pytest.mark.parametrize("lift", _LIFTS, ids=lambda c: c.__name__)
def test_a_staged_sand_under_a_lift_still_builds(lift: type, sand: str) -> None:
    """Only the direct inner aborts: ``StagedStrain`` answers ``getType``
    itself, and the probe runs the pair to the end at the pin."""
    mat = lift(inner=StagedStrain(inner=_leaf(sand)))
    assert isinstance(mat.inner, StagedStrain)


def test_init_def_grad_over_an_isotropic_lift_still_builds() -> None:
    mat = InitDefGrad(inner=LogStrain(inner=_leaf("ElasticIsotropic")))
    assert isinstance(mat.inner, LogStrain)


@pytest.mark.parametrize("sand", _SANDS)
def test_init_def_grad_directly_over_a_sand_is_left_to_the_fork(
    sand: str,
) -> None:
    """The fork refuses a small-strain inner with an error, not an exit,
    so the bridge does not add a refusal of its own (refuse what aborts,
    not more)."""
    mat = InitDefGrad(inner=_leaf(sand))
    assert type(mat.inner).__name__ == sand


# ---------------------------------------------------------------------------
# Live-fork probe: the oracle for FINITE_LIFT_ABORTING_INNERS
# ---------------------------------------------------------------------------

_STAGE = "FWMD_STAGE:"

_QUAD4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
_HEX8 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

#: chain id -> (host token, ndm, host argv tail, wrapper tokens applied
#: innermost first; the element carries the outermost).
_CHAINS: dict[str, tuple[str, int, list[Any], list[str]]] = {
    "LogStrain": ("LadrunoBrick", 3, ["-geom", "finite"], ["LogStrain"]),
    "LogStrain2D": ("LadrunoQuad", 2, ["-geom", "finite", "-thick", 1.0],
                    ["LogStrain2D"]),
    "InitDefGrad-LogStrain": ("LadrunoBrick", 3, ["-geom", "finite"],
                              ["LogStrain", "InitDefGrad"]),
    "InitDefGrad": ("LadrunoBrick", 3, ["-geom", "finite"], ["InitDefGrad"]),
    "LogStrain-StagedStrain": ("LadrunoBrick", 3, ["-geom", "finite"],
                               ["StagedStrain", "LogStrain"]),
    "LogStrain2D-StagedStrain": ("LadrunoQuad", 2,
                                 ["-geom", "finite", "-thick", 1.0],
                                 ["StagedStrain", "LogStrain2D"]),
    "LogStrain2D-PlaneStrain": ("LadrunoQuad", 2,
                                ["-geom", "finite", "-thick", 1.0],
                                ["PlaneStrain", "LogStrain2D"]),
}

_LEAVES = (*_SANDS, "ElasticIsotropic")

#: What each pair did at FORK_PIN 100eb9ae (live-fork run 39883289925):
#: ``abort`` the process exits at the first wrapper line with
#: ``<inner>::getType -- subclass responsibility``; ``refused`` the fork
#: raises an OpenSeesError at a wrapper line (InitDefGrad wants a
#: finite-strain inner, LogStrain2D an order-6 one); ``runs`` the deck
#: builds the element and analyses to the end.
_EXPECTED: dict[str, str] = {
    "LogStrain": "abort",
    "LogStrain2D": "abort",
    "InitDefGrad-LogStrain": "abort",
    "InitDefGrad": "refused",
    "LogStrain-StagedStrain": "runs",
    "LogStrain2D-StagedStrain": "runs",
    "LogStrain2D-PlaneStrain": "refused",
}


def _expected(leaf: str, chain_id: str) -> str:
    want = _EXPECTED[chain_id]
    if leaf == "ElasticIsotropic" and want == "abort":
        return "runs"
    return want


def _leaf_argv(leaf: str) -> tuple[str, list[Any]]:
    """The token and argv the bridge emits for ``leaf`` (its own _emit)."""
    rec = RecordingEmitter()
    _leaf(leaf)._emit(rec, 1)
    assert len(rec.calls) == 1, rec.calls
    _verb, args, _kw = rec.calls[0]
    return str(args[0]), list(args[2:])


def child_main(leaf: str, chain_id: str) -> None:
    """Declare the chain with raw ops (past the bridge's refusal), build
    one element, run it; print a marker after each stage."""
    from apeGmsh.opensees.emitter.live import get_backend_info, get_ops

    def stage(name: str) -> None:
        print(_STAGE + name, flush=True)

    ops = get_ops()
    stage("build=" + str(get_backend_info().build))
    token, ndm, tail, chain = _CHAINS[chain_id]
    leaf_token, leaf_args = _leaf_argv(leaf)
    ops.wipe()
    ops.model("basic", "-ndm", ndm, "-ndf", ndm)
    ops.nDMaterial(leaf_token, 1, *leaf_args)
    stage("leaf")
    inner = 1
    for depth, wtoken in enumerate(chain, start=1):
        outer = 10 * depth + 1
        ops.nDMaterial(wtoken, outer, inner)
        inner = outer
        stage("declared:" + wtoken)
    coords = _HEX8 if ndm == 3 else _QUAD4
    eps = (-1e-3, -1.5e-3, 0.0)[:ndm]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    nodes = []
    for i, c in enumerate(coords):
        xyz = [float(v) for v in c]
        ops.node(i + 1, *xyz)
        nodes.append(i + 1)
        for d in range(ndm):
            ops.sp(i + 1, d + 1, eps[d] * xyz[d])
    ops.element(token, 1, *nodes, inner, *tail)
    stage("element")
    ops.constraints("Penalty", 1.0e12, 1.0e12)
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1e-10, 25, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.25)
    ops.analysis("Static")
    rc = ops.analyze(4)
    stage(f"analyzed:rc={rc}")
    ops.wipe()


def _in_child(leaf: str, chain_id: str) -> tuple[int, list[str], str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), str(_ROOT)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    code = (
        "from tests.opensees.unit import test_finite_wrapper_over_md_refused as m\n"
        f"m.child_main({leaf!r}, {chain_id!r})\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )
    stages = [ln[len(_STAGE):] for ln in done.stdout.splitlines()
              if ln.startswith(_STAGE)]
    return done.returncode, stages, done.stderr[-2000:]


def _pinned_sha() -> str:
    for line in (_ROOT / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


def _outcome(leaf: str, rc: int, stages: list[str], err: str) -> str:
    if stages and stages[-1] == "analyzed:rc=0":
        return "runs"
    if (rc != 0 and stages and stages[-1] == "leaf"
            and f"{_EXITS_AS.get(leaf, leaf)}::getType -- subclass "
                "responsibility" in err):
        return "abort"
    if rc == 1 and "OpenSeesError" in err and "element" not in stages:
        return "refused"
    return f"other(rc={rc}, last={stages[-1] if stages else '-'})"


@pytest.mark.live
@pytest.mark.ladruno_fork
@pytest.mark.parametrize("chain_id", sorted(_CHAINS))
@pytest.mark.parametrize("leaf", _LEAVES)
def test_probe_lift_over_inner_at_pin(leaf: str, chain_id: str) -> None:
    rc, stages, err = _in_child(leaf, chain_id)
    assert stages and stages[0].startswith("build="), (rc, stages, err)
    if os.environ.get("APEGMSH_FORK_PIN_ENFORCE") == "1":
        assert stages[0] == "build=" + _pinned_sha(), stages[0]
    got = _outcome(leaf, rc, stages, err)
    want = _expected(leaf, chain_id)
    hint = ""
    if want == "abort":
        hint = (" The fork no longer exits here: #2499 may be fixed; take "
                f"{leaf} off FINITE_LIFT_ABORTING_INNERS (TODO #2499) and "
                "re-pin this table.")
    assert got == want, (
        f"{leaf} under {chain_id}: {got}, pinned {want}.{hint}\n"
        f"stages={stages}\n{err}")


def test_the_gate_matches_the_probe_table() -> None:
    """The refused set is exactly the sands whose direct lift aborts."""
    aborting = {leaf for leaf in _LEAVES
                if _expected(leaf, "LogStrain") == "abort"
                or _expected(leaf, "LogStrain2D") == "abort"}
    assert aborting == set(FINITE_LIFT_ABORTING_INNERS)
