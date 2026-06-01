"""Regenerate the committed ``.ladruno`` test fixtures.

These fixtures ground the ``Results.from_ladruno`` reader work (recorder-plan
L2-L4) so apeGmsh's reader tests run **fork-free** — CI on stock ``openseespy``
reads the committed HDF5; only *regenerating* them needs the Ladruno fork build.

Run with the fork venv::

    C:/Users/nmora/venv/opensees_venv/Scripts/python.exe \
        tests/fixtures/ladruno/_generate_fixtures.py

Requires a Ladruno fork build of OpenSees (the banner lists
"Ladruno — modular HDF5 .ladruno recorder"). Verified against build
``605affeb`` (FORMAT_VERSION 1). Each model is intentionally tiny and
deterministic. Fixtures:

  * ``truss2d.ladruno``  — 2D, nodal displacement + element basicForce
                           (value channels; the L2 baseline).
  * ``beam3d.ladruno``   — 3D ElasticBeam3d on a skew axis → a non-identity
                           ``MODEL/LOCAL_AXES`` quaternion FRAME (L3 orientation).
  * ``energy.ladruno``   — transient with ``-G energy`` → ``ON_DOMAIN`` +
                           ``ON_REGIONS`` ``energyBalance`` (L4).
"""
from __future__ import annotations

import os

import openseespy.opensees as ops

HERE = os.path.dirname(os.path.abspath(__file__))


def _truss2d(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 2.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.fix(3, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.element("truss", 2, 2, 3, 1.0, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 10.0, 0.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "basicForce")
    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 0.25)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(4)
    ops.wipe()  # flush + close the recorder


def _beam3d(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, 3.0, 1.0, 2.0)  # skew axis → non-identity local frame
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.element(
        "elasticBeamColumn", 1, 1, 2, 1.0, 2e8, 8e7, 0.1, 0.1, 0.1, 1
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "localForce")
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    ops.wipe()


def _energy(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1)
    ops.mass(2, 1.0, 1.0)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.region(1, "-node", 2)
    ops.timeSeries("Linear", 1, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 10.0, 0.0)
    # -G energy <regionTag> → whole-domain ON_DOMAIN + per-region ON_REGIONS
    ops.recorder("ladruno", path, "-N", "displacement", "-G", "energy", 1)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGen")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Linear")
    ops.analysis("Transient")
    ops.analyze(5, 0.01)
    ops.wipe()


def main() -> None:
    for name, fn in (
        ("truss2d.ladruno", _truss2d),
        ("beam3d.ladruno", _beam3d),
        ("energy.ladruno", _energy),
    ):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            os.remove(path)
        fn(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"{name}: {size} bytes")


if __name__ == "__main__":
    main()
