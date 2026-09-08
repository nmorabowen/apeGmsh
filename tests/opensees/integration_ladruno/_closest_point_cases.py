"""Child-process cases for ``test_asdplastic_closest_point_live.py``
(fork ADR-97 — the ``Closest_Point`` / ``Algorithmic`` opt-in pair).

Same contract as ``_asdplastic_cases.py``: run as
``python _closest_point_cases.py <case>`` in a fresh interpreter with
``stdin`` closed, print one ``RESULT {json}`` line.  The fork's ``.pyd``
writes ``cout`` / ``opserr`` to the process streams, which pytest's
``capfd`` cannot see but a subprocess pipe can.

The driver and the material reference set are IMPORTED from
``_asdplastic_cases`` rather than restated, so a ``Closest_Point`` run
and the ``Backward_Euler`` run it is compared against are the same
problem by construction, not by two copies that can drift apart.

Every case here needs a fork build at or after ``7e93e4381``
(``ASDP_CLOSEST_POINT_MIN_BUILD``).  On an older build the ADR-94 parser
refuses ``integration_method Closest_Point`` as an unknown value; the
``probe`` case exists to detect exactly that so the parent can SKIP
rather than fail.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import warnings

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

from _asdplastic_cases import (  # noqa: E402
    MC,
    MC_IV,
    _build_cube,
    _drive,
    _emit,
    _f_mc,
)

from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

#: The leg ``case_schema_accepted`` drives, reused verbatim.  It was
#: chosen there for three DISTINCT principal stresses; measured on fork
#: ``ff47275fd``, that holds for ``Backward_Euler`` but NOT for
#: ``Closest_Point``, which pins sxx == syy exactly -- the true
#: triaxial-compression corner.  ``Backward_Euler``'s split
#: (-3217.86 / -3244.53) is its corner-rounding error under
#: ``MC_ds=1e-5``, not physics, so the two maps are compared on the
#: yield surface and on szz, never on an sxx != syy ordering.
LEG = dict(eps_x=0.002, eps_y=0.003, eps_z=-0.01)
NSTEPS = 20

#: apeGmsh defaults ``strict_convergence=True`` (ADR 0105 D2), which
#: REFUSES this leg under ``Closest_Point``: the map converges its
#: committed states to |f| ~ 8e-13, but a TRIAL state mid-global-Newton
#: reaches f = 1.37e-06 against the 1e-06 ABSOLUTE ``f_absolute_tol``,
#: and strict turns that into a rejected step (measured: step 1 of 20,
#: ``analyze() == -3``).  At this deck's strength scale
#: (c*cos(phi) = 86.6) that residual is 1.6e-08 RELATIVE -- converged by
#: any reasonable measure.  The battery therefore runs the comparison
#: with strict OFF and pins the interaction separately in
#: ``case_strict_refuses_closest_point``; see ADR 0107.
CP = dict(
    integration_method="Closest_Point",
    tangent_type="Algorithmic",
    strict_convergence=False,
)


def _run_leg(**material_kw) -> dict:
    """Drive the shared deviatoric leg with one MohrCoulombSoil variant."""
    def make(ops):
        return ops.nDMaterial.MohrCoulombSoil(
            c=MC["c"], phi=MC["phi"], psi=MC["psi"], E=MC["E"], nu=MC["nu"],
            **material_kw,
        )

    ops = _build_cube(make, element="LadrunoBrick", nsteps=NSTEPS, **LEG)
    em, warns = _emit(ops)
    codes, hist, iters = _drive(em, NSTEPS)
    return {
        "codes": codes,
        "iters": iters,
        "warnings": warns,
        "f": [_f_mc(s, c=MC["c"], phi_deg=MC["phi"]) for s in hist],
        "last_stress": hist[-1] if hist else None,
    }


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def case_probe() -> dict:
    """Does THIS build know ``Closest_Point``?  The skip gate.

    A capability probe, not a build-hash comparison: a bare hash cannot
    prove ancestry (the same reason ``ASDP_CLOSEST_POINT_MIN_BUILD`` is
    documented and not enforced).  On a pre-ADR-97 build the ADR-94
    parser refuses the unknown value and the binding raises.
    """
    em = LiveOpsEmitter(wipe=True)
    o = em.ops
    o.model("basic", "-ndm", 3, "-ndf", 3)
    try:
        o.nDMaterial(
            "ASDPlasticMaterial3D", 1,
            "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL", MC_IV,
            "Begin_Internal_Variables",
            "BackStress", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "End_Internal_Variables",
            "Begin_Model_Parameters",
            "YoungsModulus", MC["E"], "PoissonsRatio", MC["nu"],
            "MC_phi", MC["phi"], "MC_c", MC["c"], "MC_ds", 1e-5,
            "MC_psi", MC["psi"], "MassDensity", 0.0, "InitialP0", 0.0,
            "End_Model_Parameters",
            "Begin_Integration_Options",
            "integration_method", "Closest_Point",
            "tangent_type", "Algorithmic",
            "End_Integration_Options",
        )
    except Exception as exc:                       # the binding raises
        sys.stdout.flush()
        sys.stderr.flush()
        return {
            "supported": False,
            "exception": f"{type(exc).__name__}: {exc}",
        }
    return {"supported": True, "exception": None}


def case_closest_point_leg() -> dict:
    """The ADR-97 pair converges the same leg the ADR-94 default does, to
    the same committed stress.

    Both legs run in THIS process against the same driver, so the
    comparison is of two maps on one problem.  ADR-97 D1's byte-identity
    promise is about the ``Backward_Euler`` leg being unchanged by the
    upgrade; the two maps are not required to agree bit for bit, only to
    land on the same yield surface at the same state.

    Measured on ``ff47275fd``: ``Closest_Point`` commits to |f| ~ 8e-13
    against ``Backward_Euler``'s ~1e-08 -- about four orders of magnitude
    tighter, which is the point of an exact return map.
    """
    cp = _run_leg(**CP)
    be = _run_leg()                                # helper defaults
    return {
        "closest_point": cp,
        "backward_euler": be,
        "f_absolute_tol": 1e-6,
        "strength_scale": MC["c"] * math.cos(math.radians(MC["phi"])),
    }


def case_strict_refuses_closest_point() -> dict:
    """apeGmsh's ``strict_convergence=True`` default REFUSES this leg
    under ``Closest_Point`` -- the ADR 0107 interaction, pinned.

    Not a fork defect and not an apeGmsh defect on its own: the fork
    checks the yield residual of a TRIAL state against the ABSOLUTE
    ``f_absolute_tol``, and ADR 0105 turned that check into a hard
    refusal by defaulting strict ON.  At kPa soil scale 1e-06 absolute is
    ~1.6e-08 relative, tight enough that the more accurate map trips it
    where the less accurate one happens not to.
    """
    strict = _run_leg(
        integration_method="Closest_Point", tangent_type="Algorithmic",
    )                                              # strict defaults True
    relaxed = _run_leg(**CP)
    return {
        "strict_codes": strict["codes"],
        "relaxed_codes": relaxed["codes"],
        "strict_max_f": max((abs(v) for v in strict["f"]), default=None),
        "relaxed_max_f": max((abs(v) for v in relaxed["f"]), default=None),
    }


def case_cp_iterations() -> dict:
    """``cp_iterations`` reads 0 while elastic and 1 once MC yields.

    ADR 0107 D5 / the reader's ``material.cp_iterations`` bucket entry.
    Recorded through a real ``.ladruno`` file and read back through
    ``Results``, so this covers the recorder token, the bucket shape and
    the reader table in one pass -- not just the HDF5 bytes.

    The leg is the shared one scaled to 0.05, which the fork was measured
    to carry elastic for two steps and then yield: the transition is what
    makes the value meaningful, so both halves are asserted.
    """
    import tempfile

    from apeGmsh.results import Results

    out = os.path.join(tempfile.mkdtemp(), "cp_iters.ladruno")
    scale, nsteps = 0.05, 4

    def make(ops):
        return ops.nDMaterial.MohrCoulombSoil(
            c=MC["c"], phi=MC["phi"], psi=MC["psi"], E=MC["E"], nu=MC["nu"],
            **CP,
        )

    ops = _build_cube(
        make, element="LadrunoBrick", nsteps=nsteps,
        **{k: v * scale for k, v in LEG.items()},
    )
    ops.recorder.Ladruno(
        file=out, elem_responses=("stress", "material.cp_iterations"),
    )
    em, warns = _emit(ops)
    codes, hist, _ = _drive(em, nsteps)
    em.ops.remove("recorders")                     # flush the .ladruno

    with warnings_recorded() as read_warns:
        res = Results.from_ladruno(out)
        available = sorted(res.elements.gauss.available_components())
    return {
        "codes": codes,
        "warnings": warns,
        "available": available,
        "read_warnings": read_warns,
        "per_step": _cp_per_step(out),
        "f_last": _f_mc(hist[-1], c=MC["c"], phi_deg=MC["phi"]) if hist else None,
    }


@contextlib.contextmanager
def warnings_recorded():
    """Collect warning class+message strings raised inside the block."""
    out: list[str] = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        yield out
    out.extend(f"{w.category.__name__}: {w.message}" for w in rec)


def _cp_per_step(path: str) -> list[list[float]]:
    """Distinct cp_iterations values per recorded step, straight from H5."""
    import h5py
    import numpy as np

    key = (
        "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/material.cp_iterations/"
        "33002-LadrunoBrick[0:0:0]/DATA"
    )
    with h5py.File(path, "r") as f:
        d = np.asarray(f[key])
    return [sorted(set(d[i, 0, :].tolist())) for i in range(d.shape[0])]


def case_algorithmic_without_cp_refused() -> dict:
    """``tangent_type Algorithmic`` with ``Backward_Euler`` — refused on
    BOTH sides of the seam.

    Client side: apeGmsh's ADR-97 D2 cross-check raises before any fork
    process is touched.  Server side: the same pairing pushed through the
    generic primitive is refused by the fork's own parser, and that text
    (not a reimplementation of it) is what a user sees.
    """
    from apeGmsh.opensees.material.nd import MohrCoulombSoil

    client = None
    try:
        MohrCoulombSoil(
            c=MC["c"], phi=MC["phi"], psi=MC["psi"], E=MC["E"], nu=MC["nu"],
            tangent_type="Algorithmic",
        )
    except ValueError as exc:
        client = str(exc)

    em = LiveOpsEmitter(wipe=True)
    o = em.ops
    o.model("basic", "-ndm", 3, "-ndf", 3)
    server_refused = False
    server_exc = None
    try:
        o.nDMaterial(
            "ASDPlasticMaterial3D", 1,
            "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL", MC_IV,
            "Begin_Internal_Variables",
            "BackStress", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "End_Internal_Variables",
            "Begin_Model_Parameters",
            "YoungsModulus", MC["E"], "PoissonsRatio", MC["nu"],
            "MC_phi", MC["phi"], "MC_c", MC["c"], "MC_ds", 1e-5,
            "MC_psi", MC["psi"], "MassDensity", 0.0, "InitialP0", 0.0,
            "End_Model_Parameters",
            "Begin_Integration_Options",
            "integration_method", "Backward_Euler",
            "tangent_type", "Algorithmic",
            "End_Integration_Options",
        )
    except Exception as exc:
        server_refused = True
        server_exc = f"{type(exc).__name__}: {exc}"
    sys.stdout.flush()
    sys.stderr.flush()
    return {
        "client_message": client,
        "server_refused": server_refused,
        "server_exception": server_exc,
    }


CASES = {
    name[len("case_"):]: fn
    for name, fn in list(globals().items()) if name.startswith("case_")
}


def main(argv: list[str]) -> int:
    name = argv[1]
    em = LiveOpsEmitter(wipe=True)
    build = em.ops.ladrunoBuild()
    print(f"LADRUNO_BUILD {build}")
    result = CASES[name]()
    result["ladruno_build"] = build
    sys.stdout.flush()
    print("RESULT " + json.dumps(result))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
