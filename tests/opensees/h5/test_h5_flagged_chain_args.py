"""H5 round-trip of analysis-chain args that mix flags with values.

``system Pardiso -matrixType 1 -krylov 6 -stats`` (fork ADR-75) and
``integrator Newmark 0.5 0.25 -form D`` both reach ``H5Emitter.system`` /
``.integrator`` as a tuple mixing strings with numbers.  ``_set_attr``
had no dtype for that shape and fell through to ``[float(v) for v in
value]``, so ``ops.h5(...)`` raised ``ValueError: could not convert
string to float: '-matrixType'`` — any optioned solver was unarchivable.

The proof is deck-equality *with types intact*: the archive must re-emit
``-matrixType 1``, not ``-matrixType 1.0``, because the fork parses that
value with ``OPS_GetIntInput`` and a float fails the parse.
"""
from __future__ import annotations

from pathlib import Path

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.h5.test_h5_stages_reader import build_two_quad_fem


def _bridge(system_kwargs: "dict[str, object]") -> apeSees:
    """Two-quad model whose chain carries flag-and-value args in both
    the ``system`` and the ``integrator`` slot."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    ops.constraints.Plain()
    ops.numberer.RCM()
    getattr(ops.system, str(system_kwargs.pop("_name")))(**system_kwargs)
    ops.test.NormDispIncr(tol=1e-4, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.Newmark(gamma=0.5, beta=0.25, form="D")
    ops.analysis.Transient()
    return ops


def _chain_lines(deck: str) -> list[str]:
    return [
        ln for ln in deck.splitlines()
        if ln.startswith(("system ", "integrator "))
    ]


def _round_trip(system_kwargs: "dict[str, object]", tmp_path: Path) -> None:
    tcl = tmp_path / "bridge.tcl"
    _bridge(dict(system_kwargs)).tcl(str(tcl), progress=False)
    expected = _chain_lines(tcl.read_text(encoding="utf-8"))

    h5 = tmp_path / "model.h5"
    _bridge(dict(system_kwargs)).h5(str(h5))
    replayed = _chain_lines(OpenSeesModel.from_h5(str(h5)).build("tcl"))

    # Byte-exact, so an int that came back as a float is a failure.
    assert replayed == expected


def test_pardiso_full_option_house_round_trips(tmp_path: Path) -> None:
    _round_trip(
        {"_name": "Pardiso", "matrix_type": "spd", "krylov": 6,
         "stats": True},
        tmp_path,
    )


def test_mumps_mixed_int_and_float_options_round_trip(tmp_path: Path) -> None:
    """``-ICNTL14 200`` (int) beside ``-CNTL7 1e-09`` (float) — the
    token encoding has to keep them apart, not flatten both to float."""
    _round_trip(
        {"_name": "Mumps", "icntl14": 200, "matrix_type": "symmetric",
         "cntl7": 1e-9, "stats": True},
        tmp_path,
    )


def test_bare_solver_still_round_trips(tmp_path: Path) -> None:
    """No options → no ``*_args`` attr at all; the Newmark ``-form``
    tuple still exercises the mixed path on its own."""
    _round_trip({"_name": "UmfPack"}, tmp_path)
