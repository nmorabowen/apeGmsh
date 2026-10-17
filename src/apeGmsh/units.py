"""Consistent unit system declarations and dimensional conversions.

Every finite-element solver in apeGmsh (Gmsh, OpenSees) is unitless: a
model is physically consistent only when all lengths, forces, masses,
times, and derived quantities are expressed in one coherent ``(F, L, T)``
system where ``M = F * T**2 / L``.

This module defines the seven canonical consistent unit systems shared
with ``baseUnits`` and ``apeGmshViewer/src/state/units.ts``:

======================  ==================  ================  ========  ========
Canonical ID            ``baseUnits.BASE``  Force (SI ``N``)  Length    Pressure
======================  ==================  ================  ========  ========
``"kN_m_s"``            ``kN-m-tonne-s``    ``1000.0``        ``m``     ``kPa``
``"tf_m_s"``            ``tf-m-s``          ``9806.65``       ``m``     ``tf/m²``
``"kgf_m_s"``           ``kgf-m-s``         ``9.80665``       ``m``     ``kgf/m²``
``"N_mm_s"``            ``N-mm-tonne-s``    ``1.0``           ``mm``    ``MPa``
``"N_m_s"``             ``N-m-kg-s``        ``1.0``           ``m``     ``Pa``
``"kip_in_s"``          ``kip-in-s``        ``4448.2216152605`` ``in``  ``ksi``
``"dyne_cm_s"``         ``dyne-cm-g-s``     ``1e-5``          ``cm``    ``dyne/cm²``
======================  ==================  ================  ========  ========

Use :func:`resolve_unit_system` to validate and normalize a user's unit
declaration (a canonical string ID, a ``baseUnits.BASE`` label, a
``baseUnits.systems.*`` module, or a ``baseUnits.make_system(...)``
namespace) into its canonical :data:`UnitSystemId`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "DIM_SPECS",
    "DimensionExponents",
    "FieldDimensionKind",
    "FieldUnitsResolution",
    "UNIT_SYSTEM_IDS",
    "UNIT_SYSTEMS",
    "UnitNotDeclaredWarning",
    "UnitSystemId",
    "UnitSystemInput",
    "UnitSystemSpec",
    "classify_field_dimension",
    "conversion_factor",
    "resolve_field_units",
    "resolve_unit_system",
    "unit_spec",
    "unit_symbol_for",
]

UnitSystemId = Literal[
    "kN_m_s",
    "tf_m_s",
    "kgf_m_s",
    "N_mm_s",
    "N_m_s",
    "kip_in_s",
    "dyne_cm_s",
]

FieldDimensionKind = Literal[
    "dimensionless",
    "angle",
    "frequency",
    "length",
    "curvature",
    "velocity",
    "acceleration",
    "force",
    "moment",
    "energy",
    "shell_force_per_length",
    "shell_moment_per_length",
    "pressure",
]


class UnitNotDeclaredWarning(UserWarning):
    """Issued when an ``apeGmsh`` user session is created without ``units=``.

    Loosely enforces declaring a consistent unit system at the start of a
    model script so ``model.h5``, ``.ladruno`` provenance, ``Results``,
    and ``apeGmshViewer`` know the native units without guessing.
    """


@dataclass(frozen=True)
class UnitSystemSpec:
    """Specification of a consistent ``(F, L, T)`` unit system."""

    id: UnitSystemId
    label: str
    F: float
    L: float
    T: float
    force_symbol: str
    length_symbol: str
    time_symbol: str
    pressure_symbol: str

    @property
    def base_label(self) -> str:
        return self.label

    @property
    def BASE(self) -> str:
        """``baseUnits``-compatible system label (alias for :attr:`label`)."""
        return self.label

    @property
    def M(self) -> float:
        """Derived base mass in SI kilograms (``M = F * T**2 / L``)."""
        return self.F * (self.T ** 2) / self.L

    @property
    def N(self) -> float:
        """``baseUnits``-compatible SI-newton scale factor (``1.0 / F``)."""
        return 1.0 / self.F

    @property
    def m(self) -> float:
        """``baseUnits``-compatible SI-metre scale factor (``1.0 / L``)."""
        return 1.0 / self.L

    @property
    def s(self) -> float:
        """``baseUnits``-compatible SI-second scale factor (``1.0 / T``)."""
        return 1.0 / self.T

    @property
    def kg(self) -> float:
        """``baseUnits``-compatible SI-kilogram scale factor (``1.0 / M``)."""
        return 1.0 / self.M

    @property
    def symbols(self) -> dict[str, str]:
        return {
            "force": self.force_symbol,
            "length": self.length_symbol,
            "time": self.time_symbol,
            "pressure": self.pressure_symbol,
        }


UNIT_SYSTEM_IDS: tuple[UnitSystemId, ...] = (
    "kN_m_s",
    "tf_m_s",
    "kgf_m_s",
    "N_mm_s",
    "N_m_s",
    "kip_in_s",
    "dyne_cm_s",
)

UNIT_SYSTEMS: dict[UnitSystemId, UnitSystemSpec] = {
    "kN_m_s": UnitSystemSpec(
        id="kN_m_s",
        label="kN-m-tonne-s",
        F=1e3,
        L=1.0,
        T=1.0,
        force_symbol="kN",
        length_symbol="m",
        time_symbol="s",
        pressure_symbol="kPa",
    ),
    "tf_m_s": UnitSystemSpec(
        id="tf_m_s",
        label="tf-m-s",
        F=9806.65,
        L=1.0,
        T=1.0,
        force_symbol="tf",
        length_symbol="m",
        time_symbol="s",
        pressure_symbol="tf/m²",
    ),
    "kgf_m_s": UnitSystemSpec(
        id="kgf_m_s",
        label="kgf-m-s",
        F=9.80665,
        L=1.0,
        T=1.0,
        force_symbol="kgf",
        length_symbol="m",
        time_symbol="s",
        pressure_symbol="kgf/m²",
    ),
    "N_mm_s": UnitSystemSpec(
        id="N_mm_s",
        label="N-mm-tonne-s",
        F=1.0,
        L=1e-3,
        T=1.0,
        force_symbol="N",
        length_symbol="mm",
        time_symbol="s",
        pressure_symbol="MPa",
    ),
    "N_m_s": UnitSystemSpec(
        id="N_m_s",
        label="N-m-kg-s",
        F=1.0,
        L=1.0,
        T=1.0,
        force_symbol="N",
        length_symbol="m",
        time_symbol="s",
        pressure_symbol="Pa",
    ),
    "kip_in_s": UnitSystemSpec(
        id="kip_in_s",
        label="kip-in-s",
        F=4448.2216152605,
        L=0.0254,
        T=1.0,
        force_symbol="kip",
        length_symbol="in",
        time_symbol="s",
        pressure_symbol="ksi",
    ),
    "dyne_cm_s": UnitSystemSpec(
        id="dyne_cm_s",
        label="dyne-cm-g-s",
        F=1e-5,
        L=1e-2,
        T=1.0,
        force_symbol="dyne",
        length_symbol="cm",
        time_symbol="s",
        pressure_symbol="dyne/cm²",
    ),
}

#: Mapping of accepted string aliases (canonical IDs, ``baseUnits.BASE``
#: labels, and hyphenated short forms) to canonical :data:`UnitSystemId`.
_STRING_ALIASES: dict[str, UnitSystemId] = {
    "kN_m_s": "kN_m_s",
    "kN-m-s": "kN_m_s",
    "kN-m-tonne-s": "kN_m_s",
    "tf_m_s": "tf_m_s",
    "tf-m-s": "tf_m_s",
    "kgf_m_s": "kgf_m_s",
    "kgf-m-s": "kgf_m_s",
    "N_mm_s": "N_mm_s",
    "N-mm-s": "N_mm_s",
    "N-mm-tonne-s": "N_mm_s",
    "N_m_s": "N_m_s",
    "N-m-s": "N_m_s",
    "N-m-kg-s": "N_m_s",
    "SI": "N_m_s",
    "si": "N_m_s",
    "kip_in_s": "kip_in_s",
    "kip-in-s": "kip_in_s",
    "kip-inches-s": "kip_in_s",
    "dyne_cm_s": "dyne_cm_s",
    "dyne-cm-s": "dyne_cm_s",
    "dyne-cm-g-s": "dyne_cm_s",
    "dyne-cm-gram-s": "dyne_cm_s",
}

UnitSystemInput = UnitSystemId | str | UnitSystemSpec | Any


def resolve_unit_system(units: UnitSystemInput | None) -> UnitSystemId | None:
    """Validate and normalize ``units`` to a canonical :data:`UnitSystemId`.

    Accepts:
    - ``None`` → returns ``None``.
    - A canonical ID or ``baseUnits.BASE`` string (e.g. ``"kN_m_s"``,
      ``"kN-m-tonne-s"``, ``"tf_m_s"``).
    - A :class:`UnitSystemSpec` instance.
    - A ``baseUnits.systems.*`` module or ``baseUnits.make_system(...)``
      namespace exposing ``BASE`` and/or ``(N, m, s[, kg])`` float scale
      factors. When ``(N, m, s, kg)`` are present, verifies consistency
      ``F = M * L / T**2`` before matching.

    Raises
    ------
    ValueError
        If ``units`` is not a recognized consistent unit system or if a
        custom namespace has inconsistent ``(F, L, M, T)`` scale factors.
    TypeError
        If ``units`` is not a string, :class:`UnitSystemSpec`, or
        ``baseUnits`` module/namespace.
    """
    if units is None:
        return None
    if isinstance(units, UnitSystemSpec):
        if units.id not in UNIT_SYSTEMS:
            raise ValueError(
                f"Unknown UnitSystemSpec id {units.id!r}; expected one of "
                f"{UNIT_SYSTEM_IDS}."
            )
        return units.id
    if isinstance(units, str):
        key = units.strip()
        if key in _STRING_ALIASES:
            return _STRING_ALIASES[key]
        raise ValueError(
            f"Unknown unit system {units!r}. Expected one of "
            f"{UNIT_SYSTEM_IDS} (or a baseUnits.systems module)."
        )

    # Check if it's a baseUnits module or SimpleNamespace
    has_factors = all(hasattr(units, attr) for attr in ("N", "m", "s"))
    base_attr = getattr(units, "BASE", None)
    if not has_factors and not isinstance(base_attr, str):
        raise TypeError(
            f"Invalid units declaration {units!r} (type {type(units).__name__}): "
            f"expected a canonical unit system string in {UNIT_SYSTEM_IDS} or a "
            "baseUnits system module/namespace."
        )

    if has_factors:
        try:
            n_val = float(getattr(units, "N"))
            m_val = float(getattr(units, "m"))
            s_val = float(getattr(units, "s"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unit system object {units!r} has non-numeric N/m/s factors."
            ) from exc
        if n_val <= 0.0 or m_val <= 0.0 or s_val <= 0.0:
            raise ValueError(
                f"Unit system object {units!r} has non-positive N/m/s factors."
            )
        # In baseUnits, sys.N = 1.0 / F_SI, sys.m = 1.0 / L_SI, sys.s = 1.0 / T_SI
        f_si = 1.0 / n_val
        l_si = 1.0 / m_val
        t_si = 1.0 / s_val
        if hasattr(units, "kg"):
            kg_val = float(getattr(units, "kg"))
            if kg_val <= 0.0:
                raise ValueError(
                    f"Unit system object {units!r} has non-positive kg factor."
                )
            m_si = 1.0 / kg_val
            derived_m_si = f_si * (t_si ** 2) / l_si
            if not math.isclose(m_si, derived_m_si, rel_tol=1e-9):
                raise ValueError(
                    f"Inconsistent unit system {units!r}: F={f_si} N, L={l_si} m, "
                    f"T={t_si} s implies M={derived_m_si} kg, but M={m_si} kg."
                )
        for sys_id, spec in UNIT_SYSTEMS.items():
            if (
                math.isclose(f_si, spec.F, rel_tol=1e-9)
                and math.isclose(l_si, spec.L, rel_tol=1e-9)
                and math.isclose(t_si, spec.T, rel_tol=1e-9)
            ):
                return sys_id

    if isinstance(base_attr, str) and base_attr.strip() in _STRING_ALIASES:
        return _STRING_ALIASES[base_attr.strip()]

    raise ValueError(
        f"Unrecognized consistent unit system {units!r} "
        f"(BASE={base_attr!r}). Supported systems: {UNIT_SYSTEM_IDS}."
    )


def unit_spec(units: UnitSystemInput) -> UnitSystemSpec:
    """Return the :class:`UnitSystemSpec` for ``units`` (raises on ``None``)."""
    resolved = resolve_unit_system(units)
    if resolved is None:
        raise ValueError("unit_spec requires a non-None unit system.")
    return UNIT_SYSTEMS[resolved]


@dataclass(frozen=True)
class DimensionExponents:
    """Exponents ``(force, length, time)`` in ``F**force * L**length * T**time``."""

    kind: FieldDimensionKind
    force: int
    length: int
    time: int


DIM_SPECS: dict[FieldDimensionKind, DimensionExponents] = {
    "dimensionless": DimensionExponents("dimensionless", 0, 0, 0),
    "angle": DimensionExponents("angle", 0, 0, 0),
    "frequency": DimensionExponents("frequency", 0, 0, -1),
    "length": DimensionExponents("length", 0, 1, 0),
    "curvature": DimensionExponents("curvature", 0, -1, 0),
    "velocity": DimensionExponents("velocity", 0, 1, -1),
    "acceleration": DimensionExponents("acceleration", 0, 1, -2),
    "force": DimensionExponents("force", 1, 0, 0),
    "moment": DimensionExponents("moment", 1, 1, 0),
    "energy": DimensionExponents("energy", 1, 1, 0),
    "shell_force_per_length": DimensionExponents("shell_force_per_length", 1, -1, 0),
    "shell_moment_per_length": DimensionExponents("shell_moment_per_length", 1, 0, 0),
    "pressure": DimensionExponents("pressure", 1, -2, 0),
}


def classify_field_dimension(
    field: str,
    source: str = "nodal",
    stage_kind: str | None = None,
    *,
    level: str | None = None,
) -> DimensionExponents:
    """Classify a result field name into its physical dimension in ``(F, L, T)``.

    Matches ``apeGmshViewer/src/state/units.ts::classifyFieldDimension``
    1-to-1 and extends coverage to Python ``Results`` levels (``line``,
    ``gauss``, ``fiber``, ``layer``, ``spring``, ``energy``). Mode shapes
    (``stage_kind == "mode"``) are unscaled eigenvectors and stay
    dimensionless.
    """
    if stage_kind == "mode":
        return DIM_SPECS["dimensionless"]
    eff_level = (level if level is not None else source).lower()
    name = field.lower()
    if eff_level == "energy":
        if name in ("err", "error", "error_pct", "imbalance_ratio"):
            return DIM_SPECS["dimensionless"]
        return DIM_SPECS["energy"]
    if name in (
        "kinetic",
        "elastic_strain",
        "plastic_dissipation",
        "damping_dissipation",
        "external_work",
        "imbalance",
    ) or name.endswith("_energy"):
        return DIM_SPECS["energy"]
    if name == "rotation" or name.startswith("rotation_"):
        return DIM_SPECS["angle"]
    if name.endswith("_hz") or name == "frequency":
        return DIM_SPECS["frequency"]
    if (
        name.startswith("kappa_")
        or name in ("kappa", "ky", "kz")
        or (eff_level == "line" and name.startswith("curvature"))
    ):
        return DIM_SPECS["curvature"]
    if (
        name == "footfall_r"
        or name.startswith("footfall_r_")
        or name == "footfall_utilization"
        or name.startswith("strain")
        or name.startswith("fiber_strain")
        or name.startswith("damage")
        or name.startswith("curvature")
        or name.startswith("section_deformation")
        or name in ("eps0", "gamma_y", "gamma_z")
    ):
        return DIM_SPECS["dimensionless"]
    if (
        name == "displacement"
        or name.startswith("displacement_")
        or name.startswith("spring_deformation")
        or (eff_level == "spring" and name in ("ux", "uy", "uz", "deformation"))
    ):
        return DIM_SPECS["length"]
    if eff_level == "spring" and name in ("rx", "ry", "rz"):
        return DIM_SPECS["angle"]
    if name == "velocity" or name.startswith("velocity_"):
        return DIM_SPECS["velocity"]
    if name == "acceleration" or name.startswith("acceleration_"):
        return DIM_SPECS["acceleration"]
    if (
        name == "reaction_force"
        or name.startswith("reaction_force_")
        or name in ("reaction_x", "reaction_y", "reaction_z")
        or name == "axial_force"
        or name == "shear"
        or name.startswith("shear_")
        or name in ("shear_y", "shear_z", "vy", "vz", "n", "p", "fx", "fy", "fz")
        or name == "rib_shear"
        or name.startswith("rib_shear_")
        or name == "force"
        or name.startswith("force_")
        or name.startswith("spring_force")
        or name == "local_force"
        or name.startswith("local_force_")
        or name == "basic_force"
        or name.startswith("basic_force_")
        or name == "global_force"
        or name.startswith("global_force_")
    ):
        return DIM_SPECS["force"]
    if (
        name == "reaction_moment"
        or name.startswith("reaction_moment_")
        or name in ("reaction_mx", "reaction_my", "reaction_mz")
        or name == "torsion"
        or name in ("moment_x", "moment_y", "moment_z", "mx", "my", "mz", "t")
        or name.startswith("moment_")
        or name.startswith("local_moment_")
        or name.startswith("global_moment_")
        or name == "rib_moment_pos"
        or name == "rib_moment_neg"
        or name.startswith("rib_moment_")
    ):
        return DIM_SPECS["moment"]
    if (
        name == "wood_armer_pos"
        or name == "wood_armer_neg"
        or name.startswith("wood_armer_")
        or name in ("m11", "m22", "m12")
    ):
        return DIM_SPECS["shell_moment_per_length"]
    if (
        name == "membrane_force"
        or name.startswith("membrane_force_")
        or name == "transverse_shear"
        or name.startswith("transverse_shear_")
        or name in ("f11", "f22", "f12", "q13", "q23", "v13", "v23")
    ):
        return DIM_SPECS["shell_force_per_length"]
    if name == "bending_moment" or name.startswith("bending_moment_"):
        return (
            DIM_SPECS["shell_moment_per_length"]
            if eff_level in ("element", "gauss", "layer")
            else DIM_SPECS["moment"]
        )
    if (
        name == "stress"
        or name.startswith("stress_")
        or name.startswith("fiber_stress")
        or "pressure" in name
        or name in ("von_mises", "sigma_1", "sigma_2", "sigma_3", "tau_max", "p_mean", "q_dev")
    ):
        return DIM_SPECS["pressure"]
    return DIM_SPECS["dimensionless"]


def unit_symbol_for(
    kind: FieldDimensionKind | DimensionExponents,
    sys: UnitSystemInput,
) -> str | None:
    """Format the display unit symbol for ``kind`` in ``sys``."""
    spec = unit_spec(sys)
    k: FieldDimensionKind = kind.kind if isinstance(kind, DimensionExponents) else kind
    F = spec.force_symbol
    L = spec.length_symbol
    T = spec.time_symbol
    P = spec.pressure_symbol
    if k == "dimensionless":
        return None
    if k == "angle":
        return "rad"
    if k == "frequency":
        return "Hz"
    if k == "length":
        return L
    if k == "curvature":
        return f"1/{L}"
    if k == "velocity":
        return f"{L}/{T}"
    if k == "acceleration":
        return f"{L}/{T}²"
    if k == "force":
        return F
    if k in ("moment", "energy"):
        return f"{F}·{L}"
    if k == "shell_force_per_length":
        return f"{F}/{L}"
    if k == "shell_moment_per_length":
        return f"{F}·{L}/{L}"
    if k == "pressure":
        return P
    return None


def conversion_factor(
    source_units: UnitSystemInput,
    target_units: UnitSystemInput,
    dim: DimensionExponents | FieldDimensionKind,
) -> float:
    """Return the multiplicative factor converting a quantity of ``dim``
    from ``source_units`` to ``target_units``."""
    src = unit_spec(source_units)
    dst = unit_spec(target_units)
    exp = DIM_SPECS[dim] if isinstance(dim, str) else dim
    if src.id == dst.id or (exp.force == 0 and exp.length == 0 and exp.time == 0):
        return 1.0
    return (
        ((src.F / dst.F) ** exp.force)
        * ((src.L / dst.L) ** exp.length)
        * ((src.T / dst.T) ** exp.time)
    )


@dataclass(frozen=True)
class FieldUnitsResolution:
    """Conversion factor, unit symbol, and human-readable note for a field."""

    factor: float
    unit: str | None
    note: str


def resolve_field_units(
    field: str,
    source: str,
    source_units: UnitSystemInput | None,
    target_units: UnitSystemInput | Literal["as_stored"],
    stage_kind: str | None = None,
) -> FieldUnitsResolution:
    """Compute the multiplicative conversion factor and display unit symbol
    when converting ``field`` from ``source_units`` to ``target_units``.

    Raises :class:`ValueError` when ``target_units != "as_stored"`` and
    ``source_units is None`` (user input is required to translate an
    undeclared file).
    """
    if target_units == "as_stored":
        src_id = resolve_unit_system(source_units)
        note = (
            f"Units: {UNIT_SYSTEMS[src_id].label} (declared in file; raw)."
            if src_id is not None
            else "Units: not declared in the results file."
        )
        return FieldUnitsResolution(factor=1.0, unit=None, note=note)

    src_id = resolve_unit_system(source_units)
    dst_id = resolve_unit_system(target_units)
    assert dst_id is not None
    if src_id is None:
        raise ValueError(
            f"Cannot convert field {field!r} to unit system {dst_id!r}: "
            "no unit system was declared in the model or results file. "
            "Pass source=... (e.g. results.to_units(target, source='kN_m_s')) "
            "or declare units=... on apeGmsh(...)."
        )
    src = UNIT_SYSTEMS[src_id]
    dst = UNIT_SYSTEMS[dst_id]
    dim = classify_field_dimension(field, source, stage_kind)
    factor = conversion_factor(src, dst, dim)
    unit = unit_symbol_for(dim.kind, dst)
    note = (
        f"Units: {dst.label} (file native)."
        if src.id == dst.id
        else f"Units: {dst.label} (converted from {src.label})."
    )
    return FieldUnitsResolution(factor=factor, unit=unit, note=note)
