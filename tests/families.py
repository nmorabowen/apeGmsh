"""The primitive families and the completeness exceptions (program C1.1).

Gate configuration for ``tests/test_family_completeness.py`` (panel S3,
``internal_docs/plan_expert_panel_2026-09.md``, "The family completeness
gate"). Every concrete, public ``Primitive`` subclass in
``apeGmsh.opensees`` must belong to exactly one family below and be
enumerated in that family's ``ALL_*`` contract list, or have an entry in
``EXCEPTIONS`` with a reason.

``EXCEPTIONS`` is a ratchet: it may only shrink. An entry whose class is
now covered, or no longer exists, fails the gate, so delete the entry
when you close its gap. Never add an entry for a class you are adding:
append the class to its family's ``ALL_*`` list instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from apeGmsh.opensees._internal.types import (
    Analysis,
    ConstraintHandler,
    ConvergenceTest,
    Element,
    GeomTransf,
    Integrator,
    LinearSystem,
    NDMaterial,
    Numberer,
    Pattern,
    Primitive,
    Recorder,
    Section,
    SolutionAlgorithm,
    TimeSeries,
    UniaxialMaterial,
)

_CONTRACT = "tests.opensees.contract."
_ELEMENT = "apeGmsh.opensees.element."


@dataclass(frozen=True)
class Family:
    """One primitive family and the contract list that must enumerate it.

    A class is a member when it subclasses ``base`` and, if ``modules``
    is set, is defined in one of those modules. ``modules`` splits the
    single ``Element`` base into kinds. ``contract`` is the
    ``"<module>:<ALL_NAME>"`` path of the family's ``ALL_*`` list.
    """

    name: str
    base: type[Primitive]
    contract: str
    modules: tuple[str, ...] = ()


FAMILIES: tuple[Family, ...] = (
    Family("uniaxial material", UniaxialMaterial,
           _CONTRACT + "test_uniaxial_material_contract:ALL_UNIAXIAL"),
    Family("nd material", NDMaterial,
           _CONTRACT + "test_nd_material_contract:ALL_ND"),
    Family("section", Section,
           _CONTRACT + "test_section_contract:ALL_SECTIONS"),
    Family("geomTransf", GeomTransf,
           _CONTRACT + "test_geom_transf_contract:ALL_GEOM_TRANSF"),
    Family("time series", TimeSeries,
           _CONTRACT + "test_time_series_contract:ALL_TIME_SERIES"),
    Family("pattern", Pattern,
           _CONTRACT + "test_pattern_contract:ALL_PATTERNS"),
    Family("recorder", Recorder,
           _CONTRACT + "test_recorder_contract:ALL_RECORDERS"),
    Family("beam-column element", Element,
           _CONTRACT + "test_element_beam_column_contract:ALL_BEAM_COLUMN_ELEMENTS",
           (_ELEMENT + "beam_column",)),
    Family("truss element", Element,
           _CONTRACT + "test_element_truss_contract:ALL_TRUSS_ELEMENTS",
           (_ELEMENT + "truss",)),
    Family("zero-length element", Element,
           _CONTRACT + "test_element_zero_length_contract:ALL_ZERO_LENGTH_ELEMENTS",
           (_ELEMENT + "zero_length", _ELEMENT + "two_node_link")),
    Family("shell element", Element,
           _CONTRACT + "test_element_shell_contract:ALL_SHELL_ELEMENTS",
           (_ELEMENT + "shell",)),
    Family("solid element", Element,
           _CONTRACT + "test_element_solid_contract:ALL_SOLID_ELEMENTS",
           (_ELEMENT + "solid",)),
    Family("constraint handler", ConstraintHandler,
           _CONTRACT + "test_analysis_contract:ALL_CONSTRAINT_HANDLERS"),
    Family("numberer", Numberer,
           _CONTRACT + "test_analysis_contract:ALL_NUMBERERS"),
    Family("system", LinearSystem,
           _CONTRACT + "test_analysis_contract:ALL_SYSTEMS"),
    Family("convergence test", ConvergenceTest,
           _CONTRACT + "test_analysis_contract:ALL_TESTS"),
    Family("algorithm", SolutionAlgorithm,
           _CONTRACT + "test_analysis_contract:ALL_ALGORITHMS"),
    Family("integrator", Integrator,
           _CONTRACT + "test_analysis_contract:ALL_INTEGRATORS"),
    Family("analysis", Analysis,
           _CONTRACT + "test_analysis_contract:ALL_ANALYSES"),
)


# The ratchet's ceiling: len(EXCEPTIONS) may never exceed it. Lower it
# when you close a gap. Raising it is a ratchet-baseline raise, which
# needs the maintainer (PROGRAM.md §4); never raise it to go green.
EXCEPTIONS_BASELINE = 55

# Gaps measured when the gate landed (2026-09-29). Keyed
# "<module>:<qualname>". A reason names the commit or PR that shipped the
# class without its contract entry, or why the class has no family yet.
_NO_INTEGRATION_FAMILY = (
    "no family: BeamIntegration has no contract file or ALL_* list; the "
    "rules are unit-tested in tests/opensees/unit/primitives/"
    "test_beam_integration.py (shipped 00f5619f, phase 4.5)"
)
_NO_DAMPING_FAMILY = (
    "no family: Damping (ADR 0053) has no contract file or ALL_* list; "
    "unit-tested in tests/opensees/unit/primitives/test_damping.py"
)
_NO_ABSORBING_FAMILY = (
    "no family: element.absorbing has no ALL_* list of its own; "
    "unit-tested in tests/opensees/unit/primitives/test_elements_absorbing.py"
)

EXCEPTIONS: dict[str, str] = {
    # --- no family -------------------------------------------------------
    "apeGmsh.opensees.integration:HingeEndpoint": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:HingeMidpoint": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:HingeRadau": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:HingeRadauTwo": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:Legendre": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:Lobatto": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:NewtonCotes": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:Radau": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.integration:Trapezoidal": _NO_INTEGRATION_FAMILY,
    "apeGmsh.opensees.damping.damping:Uniform":
        _NO_DAMPING_FAMILY + " (shipped 321fed81, D3a)",
    "apeGmsh.opensees.damping.damping:SecStif":
        _NO_DAMPING_FAMILY + " (shipped 321fed81, D3a)",
    "apeGmsh.opensees.damping.damping:URD":
        _NO_DAMPING_FAMILY + " (shipped a7969c11, D3b-1)",
    "apeGmsh.opensees.damping.damping:URDbeta":
        _NO_DAMPING_FAMILY + " (shipped a7969c11, D3b-1)",
    "apeGmsh.opensees.element.absorbing:ASDAbsorbingBoundary3D":
        _NO_ABSORBING_FAMILY + " (shipped d5241a70, AB-2)",
    "apeGmsh.opensees.element.absorbing:ASDAbsorbingBoundary2D":
        _NO_ABSORBING_FAMILY + " (shipped 4c1d771a, AB-5)",
    # --- nd material: missing from ALL_ND --------------------------------
    "apeGmsh.opensees.material.nd:PlaneStrain":
        "shipped 46c5c46d (SSI-1) without an ALL_ND entry; wraps a 3-D NDMaterial",
    "apeGmsh.opensees.material.nd:ASDPlasticMaterial3D":
        "shipped 46c5c46d (SSI-1) without an ALL_ND entry",
    "apeGmsh.opensees.material.nd:ASDConcrete3D":
        "shipped 6d6b3745 (ADR 0044) without an ALL_ND entry",
    "apeGmsh.opensees.material.nd:InitDefGrad":
        "shipped in #549 (Ladruno wrappers) without an ALL_ND entry; wraps an NDMaterial",
    "apeGmsh.opensees.material.nd:LogStrain":
        "shipped in #549 (Ladruno wrappers) without an ALL_ND entry; wraps an NDMaterial",
    "apeGmsh.opensees.material.nd:StagedStrain":
        "shipped in #549 (Ladruno wrappers) without an ALL_ND entry; wraps an NDMaterial",
    "apeGmsh.opensees.material.nd:LadrunoConcrete3D":
        "shipped in #726 (Ladruno concrete nD) without an ALL_ND entry",
    "apeGmsh.opensees.material.nd:LadrunoRCConcrete":
        "shipped in #726 (Ladruno concrete nD) without an ALL_ND entry",
    "apeGmsh.opensees.material.nd:LadrunoRCFiniteStrain":
        "shipped in #726 (Ladruno concrete nD) without an ALL_ND entry",
    "apeGmsh.opensees.material.nd:LadrunoCohesiveHingeBiaxial":
        "shipped in #727 (Ladruno beam-column cluster) without an ALL_ND entry",
    "apeGmsh.opensees.material.nd:LogStrain2D":
        "shipped in #1169 (2-D finite strain) without an ALL_ND entry; wraps an NDMaterial",
    "apeGmsh.opensees.material.nd:PlateRebar":
        "shipped in #1182 (shell layers) without an ALL_ND entry; wraps a UniaxialMaterial",
    "apeGmsh.opensees.material.nd:PlateFromPlaneStress":
        "shipped in #1182 (shell layers) without an ALL_ND entry; wraps an NDMaterial",
    "apeGmsh.opensees.material.nd:PlaneStressRebar":
        "shipped in #1182 (shell layers) without an ALL_ND entry; wraps a UniaxialMaterial",
    "apeGmsh.opensees.material.nd:PlateFiber":
        "shipped in #1185 (PlateFiber) without an ALL_ND entry; wraps an NDMaterial",
    # --- uniaxial material: missing from ALL_UNIAXIAL --------------------
    "apeGmsh.opensees.material.uniaxial:ASDConcrete1D":
        "shipped 6d6b3745 (ADR 0044) without an ALL_UNIAXIAL entry",
    "apeGmsh.opensees.material.uniaxial:LadrunoCohesiveHinge":
        "shipped in #727 (Ladruno beam-column cluster) without an ALL_UNIAXIAL entry",
    # --- section: missing from ALL_SECTIONS ------------------------------
    "apeGmsh.opensees.section.computed:ComputedSection":
        "shipped in #801 (ADR 0078 section analyzer) without an ALL_SECTIONS entry",
    # --- time series: missing from ALL_TIME_SERIES -----------------------
    "apeGmsh.opensees.time_series.time_series:MomentStep":
        "shipped 0a5dc09e (ADR 0062, MT-3) without an ALL_TIME_SERIES entry",
    "apeGmsh.opensees.time_series.time_series:Yoffe":
        "shipped 0a5dc09e (ADR 0062, MT-3) without an ALL_TIME_SERIES entry",
    # --- recorder: missing from ALL_RECORDERS ----------------------------
    "apeGmsh.opensees.recorder:Ladruno":
        "shipped cd284ac9 (Ladruno recorder L1) without an ALL_RECORDERS entry",
    "apeGmsh.opensees.recorder:Monitor":
        "shipped in #550 (live Monitor recorder) without an ALL_RECORDERS entry",
    # --- beam-column element: missing from ALL_BEAM_COLUMN_ELEMENTS ------
    "apeGmsh.opensees.element.beam_column:LadrunoDispBeamColumn":
        "shipped in #727 (Ladruno beam-column cluster) without an "
        "ALL_BEAM_COLUMN_ELEMENTS entry",
    "apeGmsh.opensees.element.beam_column:LadrunoIMKBeam":
        "shipped in #727 (Ladruno beam-column cluster) without an "
        "ALL_BEAM_COLUMN_ELEMENTS entry",
    # --- solid element: missing from ALL_SOLID_ELEMENTS ------------------
    "apeGmsh.opensees.element.solid:BezierTri6":
        "shipped 5c93fcf8 (Bezier B1) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:BezierTet10":
        "shipped 8c578c2c (Bezier B2) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:LadrunoBrick":
        "shipped 2a4e6672 (fork brick catch-up) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:LadrunoQuad":
        "shipped 9510d1c8 (fork plane quad) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:LadrunoCST":
        "shipped 6c881ad9 (fork plane triangle) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:LadrunoUP":
        "shipped in #793 (ADR 0074 u-p element) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:LadrunoBrick20":
        "shipped in #857 (fork second-order solids) without an ALL_SOLID_ELEMENTS entry",
    "apeGmsh.opensees.element.solid:LadrunoLST":
        "shipped in #857 (fork second-order solids) without an ALL_SOLID_ELEMENTS entry",
    # --- analysis components ---------------------------------------------
    "apeGmsh.opensees.analysis.constraint_handler:LadrunoProjection":
        "shipped in #697 (ADR 0068 tied interface) without an "
        "ALL_CONSTRAINT_HANDLERS entry",
    "apeGmsh.opensees.analysis.constraint_handler:LadrunoContact":
        "shipped in #728 (Ladruno analysis cluster) without an "
        "ALL_CONSTRAINT_HANDLERS entry",
    "apeGmsh.opensees.analysis.test:LadrunoStabilizedUnbalance":
        "shipped in #728 (Ladruno analysis cluster) without an ALL_TESTS entry",
    "apeGmsh.opensees.analysis.integrator:LadrunoGeneralizedAlpha":
        "shipped in #728 (Ladruno analysis cluster) without an ALL_INTEGRATORS entry",
    "apeGmsh.opensees.analysis.integrator:LadrunoHHT":
        "shipped in #728 (Ladruno analysis cluster) without an ALL_INTEGRATORS entry",
    "apeGmsh.opensees.analysis.integrator:CentralDifferenceSMS":
        "shipped in #730 (Ladruno SMS integrators) without an ALL_INTEGRATORS entry",
    "apeGmsh.opensees.analysis.integrator:ExplicitBatheSMS":
        "shipped in #730 (Ladruno SMS integrators) without an ALL_INTEGRATORS entry",
    "apeGmsh.opensees.analysis.integrator:ExplicitBatheLNVDSMS":
        "shipped in #730 (Ladruno SMS integrators) without an ALL_INTEGRATORS entry",
}
