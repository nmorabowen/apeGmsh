"""``ON_ELEMENTS`` decoding for :class:`LadrunoReader` (recorder plan L2b-2).

Where MPCO needs a per-class response catalog to decode element results,
the ``.ladruno`` writer is **self-describing**: every
``RESULTS/ON_ELEMENTS/<token>/<class>[…]`` bucket carries a structured
``COLUMN_MAP`` (one row per output block) plus a ``COMP_NAMES`` attribute
(newline-separated, one CSV line per block). The reader reads component
names **from the file** — this is the seam that sidesteps the per-class
GP-order traps MPCO fights.

``COLUMN_MAP`` row vocabulary (verified, fork build ``605affeb``):

* ``LEVELS`` — descriptor depth: ``0`` ElementOutput · ``1`` GaussPoint ·
  ``2`` SectionOutput · ``3`` FiberOutput · ``4`` Nd/UniaxialMaterial.
* ``GAUSS_ID`` — 0-based GP index for the block, or ``-1`` (element-level).
* ``NUM_COMP`` — components in the block (== len of its ``COMP_NAMES`` line).

``DATA`` is ``[T, nE, total_cols]`` laid out **block-major**: block *i*
occupies columns ``[Σ width_j (j<i) … +width_i)``. Tokens are mapped to
apeGmsh's neutral vocabulary here:

* continuum ``sigmaIJ`` / ``etaIJ`` → ``stress_ij`` / ``strain_ij`` (Gauss);
* beam ``N``/``Vy``/``Vz``/``T``/``My``/``Mz`` (optional ``_<station>``)
  → ``axial_force`` / ``shear_y`` / … (line stations);
* nodal ``globalForce`` columns (``Px_1``…, ``P<dof>_<node>``,
  ``P<node>_<dof>``) → ``nodal_resisting_{force,moment}_{x,y,z}``
  (elements; see :func:`decode_nodal_force_columns`).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy import ndarray

from ..._vocabulary import PER_ELEMENT_NODAL_FORCES
from ...opensees._response_catalog import (
    RESPONSE_CATALOG,
    catalog_token_for_keyword,
    is_layer_catalogued,
    local_force_station_signs,
)
from .._shell_geometry import shell_quaternion
from ._ladruno_hyperslab import read_hyperslab
from ._mpco_layer_io import (
    LayerBucketUnreadWarning,
    ShellFrameUnavailableWarning,
)

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Missing element results — fail loud, do not return empty
# =====================================================================
#
# A recorder ``-E <token>`` that matches no element writes a file with
# **no** ``RESULTS/ON_ELEMENTS`` group at all. Before this guard the
# reader answered every element / gauss / line-station / fiber read with
# an empty container, so a lost field capture looked exactly like a
# converged run with nothing to plot — a full campaign was re-run before
# anyone noticed. Engine builds from 2026-08 also warn at record time.
#
# Scope is deliberately narrow: this fires only when the file carries
# **no element results whatsoever**. A file that *has* element results
# but not the requested component (``stress_zz`` on a 2-D model,
# ``basicForce`` on a quad) keeps the long-standing empty-slab answer —
# that is a legitimate "this model has no such quantity", not data loss.


class MissingElementResults(KeyError):
    """A ``.ladruno`` carries no element results at all, yet one was read.

    Raised by :func:`require_element_results` from every element-level
    read on :class:`~apeGmsh.results.readers._ladruno.LadrunoReader`.
    Subclasses :class:`KeyError` — the read asked for a group the file
    does not have.
    """

    def __str__(self) -> str:                     # KeyError repr()s its arg
        return self.args[0] if self.args else ""


def require_element_results(
    on_elements: "Optional[h5py.Group]",
    *,
    family: str,
    component: str,
    source: object,
    stage_id: str,
) -> "h5py.Group":
    """Return ``on_elements``, or raise :class:`MissingElementResults`.

    ``family`` is the read level in user words (``"gauss (stress /
    strain)"``, ``"element"``, …), ``component`` the thing that was
    asked for, ``source`` the file path (or reader) for the message.
    Returning the group (rather than returning ``None``) keeps the
    caller's non-optional narrowing after the check.
    """
    if on_elements is not None and len(on_elements) > 0:
        return on_elements
    raise MissingElementResults(
        f"No element results in {source} (stage {stage_id!r}): the "
        f"RESULTS/ON_ELEMENTS group is "
        f"{'absent' if on_elements is None else 'empty'}, so the "
        f"requested {family} component {component!r} cannot be read.\n"
        f"Likely cause: the recorder's -E token matched no element — "
        f"check the token spelling against the element's response names "
        f"(engine builds from 2026-08 print a warning at record time). "
        f"Node-only recorders produce the same file: record an element "
        f"response if you need this level."
    )


# =====================================================================
# Token → canonical maps
# =====================================================================

_AXIS = {"1": "x", "2": "y", "3": "z"}

# Beam end / basic forces (localForce, basicForce) — element-level
# (``LEVELS==0``) buckets that carry one block of named columns.
_BEAM_BASE_TO_CANONICAL: dict[str, str] = {
    "N": "axial_force",
    "Vy": "shear_y",
    "Vz": "shear_z",
    "T": "torsion",
    "My": "bending_moment_y",
    "Mz": "bending_moment_z",
}

# Section stress-resultants (``section.force``, recorder token
# ``LEVELS==2``). One block per integration station; columns named with the
# OpenSees section response codes ``P``/``Vy``/``Vz``/``T``/``My``/``Mz``
# (``P`` is the section axial force — distinct from the element-level ``N``).
_SECTION_FORCE_TO_CANONICAL: dict[str, str] = {
    "P": "axial_force",
    "Vy": "shear_y",
    "Vz": "shear_z",
    "T": "torsion",
    "My": "bending_moment_y",
    "Mz": "bending_moment_z",
}

# Section generalized strains (``section.deformation``) — conjugate
# work-pair to the resultants, same per-station layout. Tokens are the
# OpenSees deformation codes (``eps``/``kappaZ``/…). Only the 2-D pair
# (``eps``/``kappaZ``) is verified against a fixture; the 3-D tokens follow
# the standard section response order.
_SECTION_DEFORM_TO_CANONICAL: dict[str, str] = {
    "eps": "axial_strain",
    "gammaY": "shear_strain_y",
    "gammaZ": "shear_strain_z",
    "theta": "torsional_strain",
    "kappaY": "curvature_y",
    "kappaZ": "curvature_z",
}

# Fiber-level buckets (``section.fiber.stress`` / ``section.fiber.strain``,
# ``LEVELS==4`` with ``MULTIPLICITY``==num-fibers). Keyed by token, not by
# column name (the column is the scalar uniaxial ``sigma11``/``eps11``).
#
# Two token spellings map to the same canonical: fiber-section beams emit
# under ``section.fiber.*``; layered shells emit under ``material.fiber.*``
# (the recorder swaps ``section``→``material`` for shells, recorder PR #200 —
# the bucket layout is byte-identical, only the token name differs). A model
# carrying both kinds emits both buckets, so reads gather from every present
# spelling (mirrors the MPCO reader's dual-keying, ``_mpco.py`` discover).
_FIBER_TOKEN_TO_CANONICAL: dict[str, str] = {
    "section.fiber.stress": "fiber_stress",
    "section.fiber.strain": "fiber_strain",
    "material.fiber.stress": "fiber_stress",
    "material.fiber.strain": "fiber_strain",
}
# canonical → candidate tokens, ``section.fiber.*`` first (beam convention).
_FIBER_CANONICAL_TO_TOKENS: dict[str, tuple[str, ...]] = {
    "fiber_stress": ("section.fiber.stress", "material.fiber.stress"),
    "fiber_strain": ("section.fiber.strain", "material.fiber.strain"),
}


def is_fiber_token(token: str) -> bool:
    """True for a FIBER bucket, which must never surface as continuum Gauss.

    ⚠ A TOKEN test on purpose.  ``MULTIPLICITY > 1`` looks like it would do
    the job and does not: a fiber bucket repeats one scalar column per
    fiber, but a CONTINUUM bucket repeats its component set once per GAUSS
    POINT (``LadrunoLST``: NUM_COMP 3 x MULTIPLICITY 3 = 9 columns), and
    ``SECTION_TAG`` / ``FIBER_ID`` are -1 in BOTH.  Filtering on
    multiplicity therefore discarded every solid element carrying more than
    one Gauss point -- stress / strain / eqpstrain vanished from the picker
    and the viewer drew no continuum fields at all.
    """
    return token in _FIBER_TOKEN_TO_CANONICAL or ".fiber." in token


def section_canonical(token: str) -> Optional[str]:
    """Map a ``section.force`` / ``section.deformation`` column token to a
    neutral name (``P``→``axial_force``, ``kappaZ``→``curvature_z``), else
    ``None``."""
    t = token.strip()
    return _SECTION_FORCE_TO_CANONICAL.get(t) or _SECTION_DEFORM_TO_CANONICAL.get(t)

# Continuum stress/strain tokens come in two flavours across element
# classes: the digit form ``sigma11``/``eta11`` (stock FourNodeQuad etc.;
# the Ladruno solids tag strains ``eps11``) and the axis form
# ``sigma_xx``/``eps_xx``/``gamma_xy`` (BezierTri6 and
# the elements that follow the contract's recommended naming). ``eta`` and
# ``eps``/``epsilon`` are strain; ``gamma_xy`` is engineering shear (the
# same off-diagonal the digit ``eta12`` carries) → ``strain_xy``.
_CONTINUUM_DIGIT_RE = re.compile(
    r"^(?P<kind>sigma|eta|epsp|epsilon|eps|pstrain)(?P<i>[123])(?P<j>[123])$"
)
_CONTINUUM_AXIS_RE = re.compile(
    r"^(?P<kind>sigma|epsp|epsilon|eps|gamma|pstrain|stress|strain)"
    r"_(?P<a>[xyz])(?P<b>[xyz])$"
)
# Token stem → canonical tensor root. ``epsp`` (plastic strain, the
# fork's natural tag following its ``eps11`` total-strain convention;
# ``pstrain`` is the ASDPlasticMaterial3D response-token spelling) is
# listed before ``eps`` in the alternations above so it wins the
# prefix race.
# ``stress``/``strain`` are the canonical roots themselves: blocks rebuilt
# from RESPONSE_CATALOG (see ``resolve_generic_gauss_blocks``) already carry
# apeGmsh canonicals, so canonicalisation must be idempotent on them.
_KIND_TO_ROOT = {
    "sigma": "stress",
    "eta": "strain",
    "eps": "strain",
    "epsilon": "strain",
    "gamma": "strain",
    "epsp": "plastic_strain",
    "pstrain": "plastic_strain",
    "stress": "stress",
    "strain": "strain",
}

# Scalar per-GP labels → canonical. The accumulated equivalent plastic
# strain arrives under the material's own spelling: ASDPlasticMaterial3D
# responds to ``eqpstrain``; LadrunoJ2 to ``equivalentPlasticStrain`` /
# ``plasticStrainEq`` / ``ebarP``. All map to the (already canonical,
# material-emitted) ``equivalent_plastic_strain``. Keys lowercase — the
# token is lowercased before lookup.
_CONTINUUM_SCALAR_TOKENS = {
    "eqpstrain": "equivalent_plastic_strain",
    "ebarp": "equivalent_plastic_strain",
    "equivalentplasticstrain": "equivalent_plastic_strain",
    "plasticstraineq": "equivalent_plastic_strain",
}

# ADR 0105 Amendment 1 — ASDPlasticMaterial3D's material-level buckets.
#
# Keyed by the BUCKET TOKEN (the part after ``material.``), NOT by the
# column label, and resolved by COLUMN POSITION. The labels the material
# writes are not usable as keys: ``material.PStress`` labels its one
# column ``p``, and canonicalisation is case-insensitive, so a label map
# would make every force-based beam's section axial force ``P`` resolve
# to a Gauss mean stress. The token is unambiguous — one bucket, one
# request, one known column set.
#
# Named ``material_*`` on purpose: these are what the MATERIAL computed,
# under its own definition and sign, and are NOT the reader's
# tensor-derived ``mean_stress`` / ``j2_stress`` / ``volumetric_strain``
# / ``j2_strain`` (``results/_derived.py``). Aliasing the two would need
# a per-material sign/definition audit — the fork header's comment on
# ``p`` does not match its own arithmetic — so the two provenances stay
# separately named.
#
# A tuple maps a multi-column bucket by position: ``BackStress`` is the
# material's Voigt order 11, 22, 33, 12, 23, 13 — the same order the
# ``epsP1..`` plastic-strain columns already use. Scalar internal
# variables (``iv_size == 1``) get one name. An explicit table, no
# generic pass-through: an unknown bucket must surface as a dropped
# column (:class:`GaussColumnDroppedWarning`), not as a guessed
# canonical.
#
# TIMs A12 — LadrunoSANISAND's IMPL-EX/state responses (fork PR #805 /
# #820). Same by-position resolution, and the same reason it has to be:
# older fork builds write ``C1..Cn`` instead of the named COMP_NAMES
# below, so a label map would miss those builds entirely. Each response
# is keyed once under its primary token and again under every fork
# alias (``stateparameter``, ``yieldfunction``, ``substepsme``,
# ``ladrunosubsteps``) since the recorder forwards whichever spelling
# the deck used straight through to ``-E material.<token>``.
_MATERIAL_BUCKET_TOKENS: "dict[str, str | tuple[str, ...]]" = {
    "pstress": "material_mean_stress",
    "j2stress": "material_j2_stress",
    "volstrain": "material_volumetric_strain",
    "j2strain": "material_j2_strain",
    "backstress": (
        "back_stress_xx", "back_stress_yy", "back_stress_zz",
        "back_stress_xy", "back_stress_yz", "back_stress_xz",
    ),
    "yieldstress": "yield_stress",
    "dp_cohesion": "dp_cohesion",
    # ADR 0107 / fork ADR-97 -- the local-Newton iteration count of the
    # LAST Closest_Point return-map solve at that Gauss point. A scalar
    # column per Gauss point, OBSERVED on fork build ff47275fd rather
    # than assumed: NUM_COMP 1, MULTIPLICITY 1, FIBER_ID -1, one column
    # per GAUSS_ID. Measured 0 while the point is elastic and 1 once it
    # yields on the planar families (MC / MCTC / VonMises /
    # DruckerPrager); the fork documents up to 5 for Hoek-Brown's curved
    # surface. UNDEFINED under Backward_Euler -- the fork still writes
    # the bucket, so a value read off a non-Closest_Point deck is
    # meaningless rather than absent.
    "cp_iterations": "cp_iterations",
    "cappressure": "cap_pressure",
    "epsqpshear": "eps_qp_shear",
    "psi": "state_parameter",
    "stateparameter": "state_parameter",
    "yielddistance": "yield_distance",
    "yieldfunction": "yield_distance",
    "implexerror": "implex_error",
    "avgimplexerror": "avg_implex_error",
    "substeps": ("substeps_me", "substeps_cap_hit"),
    "substepsme": ("substeps_me", "substeps_cap_hit"),
    "ladrunosubsteps": ("substeps_me", "substeps_cap_hit"),
    "implexdetail": (
        "implex_detail_total", "implex_detail_dev", "implex_detail_vol",
        "implex_detail_clamp_fired", "implex_detail_clamp_count",
        "implex_detail_f",
    ),
    # Fork PR #838 (WP-99, F7) widened this 4 -> 6. Slot 4
    # (commit_latched) is the ONLY per-instance slot -- each Gauss
    # point's own commit-refusal latch, 0/1, sticky until
    # revertToStart(). Slot 5 (latched) is process-wide but fires once
    # per NEWTON ITERATION per latched point, not once per refusal.
    # Neither is summable like slots 0-3 (0 is already their sum):
    # reduce slot 4 with max()/a latched-point count, never a sum.
    "implexrefusals": (
        "implex_refusals_total", "implex_refusals_sign_change",
        "implex_refusals_control", "implex_refusals_companion",
        "implex_refusals_commit_latched", "implex_refusals_latched",
    ),
    # ADR 92 P2-9 grew this census 6 -> 7 slots. The fork sets NO
    # ResponseType for it (LadrunoSANISAND.cpp:3947-3951 returns a bare
    # MaterialResponse), so unlike every other fork response here the file
    # always writes C1..C7 and these names are the ONLY ones there will be.
    # Order is not the adoption guide's suggested list -- it is the fill
    # site, LadrunoSANISAND.cpp:3996-4002, which is the only authority on a
    # by-position map.
    "implexguards": (
        "implex_guards_floor_fallback",     # 0  P2-1 floor -> implicit stress
        "implex_guards_f0_guard",           # 1  P2-2 f = 0 after reversal/softening
        "implex_guards_hold_preserved",     # 2  P2-3 zero-dt commits left alone
        "implex_guards_reversal_noise",     # 3  P2-5 reversal-noise guards
        "implex_guards_trial_f0_guard",     # 4  P2-6 trial-time f = 0 fallbacks
        "implex_guards_hold_skip_commit",   # 5  P2-5c hold-skip commits
        "implex_guards_control_backoff",    # 6  P2-9 f* < 0.5*f_max
    ),
    # Fork ADR-95 (PR #803). Same shape as implexGuards: the fork returns a
    # bare ``MaterialResponse(this, 95, Vector(8))`` with NO ResponseType
    # (DruckerPrager.cpp:1004-1005), so the file always writes C1..C8 and
    # this map is the only authority on what they mean. Order is the fill
    # site, ``DruckerPrager::getLadrunoBranch()``, DruckerPrager.cpp:960-970.
    # The map wins over the file's own labels whenever the widths match, so
    # re-check this entry if ``DruckerPrager::setResponse`` ever starts
    # emitting a ResponseType for responseID 95.
    # ``ladrunoTangent`` (36 entries, responseID 96) is refused bare by the
    # recorder but deliberately NOT named here -- nothing reads it yet.
    "ladrunobranch": (
        "dp_branch",            # 0  0 elastic / 1 cone / 2 cutoff / 3 corner
        "dp_gamma_cone",        # 1  gamma(0), the f1 plastic multiplier
        "dp_gamma_cutoff",      # 2  gamma(1), the f2 plastic multiplier
        "dp_f1_trial",          # 3  f1 at the TRIAL state
        "dp_f2_trial",          # 4  f2 at the TRIAL state
        "dp_forced_accept",     # 5  1 when the count > 3 bailout fired
        "dp_i1",                # 6  I1 of the RETURNED stress
        "dp_det_a_min",         # 7  min_n det(n.D_ep.n) / (2G)^3
    ),
    # The concrete nD laws' dual damage, tension first: ASDConcrete3D's
    # ``damage`` (labels ``d+``, ``d-``; ASDConcrete3DMaterial.cpp
    # setResponse id 2001) and LadrunoConcrete3D's ``damage`` / ``omega``
    # (a bare ``MaterialResponse(this, 5, Vector(2))`` -> ``C1, C2``;
    # LadrunoConcrete3D.cpp getResponse case 5 fills omegaT, omegaC).
    # The canonicals are MATERIAL_STATE's conjugate pair, the names the
    # live capture and the MPCO ``d+``/``d-`` META symbols also land on.
    "damage": ("damage_tension", "damage_compression"),
    "omega": ("damage_tension", "damage_compression"),
}
_MATERIAL_PREFIX = "material."
_BEAM_RE = re.compile(r"^(?P<base>[A-Za-z]+?)(?:_(?P<station>\d+))?$")

# Runway A1 — the fork's REAL ``COMP_NAMES`` for the tokens above, exact
# spelling, in slot order. ``_MATERIAL_BUCKET_TOKENS`` resolves by
# POSITION on purpose (older builds write generic ``C1..Cn``), which
# means a real-name build whose columns are reordered or renamed is
# mislabelled in silence — a measured probe found exactly that for
# ``material.substeps`` (file order ``substeps_capHit, substeps_me`` read
# as ``substeps_me, substeps_capHit``) and for ``material.psi`` (labelled
# ``yieldDistance``, read as ``state_parameter``). This table lets
# :func:`_name_mismatch` catch the disagreement and warn instead of
# mislabelling.
#
# Deliberately partial. Only tokens whose exact fork spelling is
# documented are listed:
# * the SANISAND tokens per TIMs A12 (fork PR #820) / the adoption guide;
# * the ADR 0105 ASDPlasticMaterial3D tokens whose spelling
#   ``internal_docs/guide_ladruno_asdplastic.md`` states explicitly —
#   ``p``, ``J2stress``, ``epsVol``, ``J2strain``, ``BackStress_1..6``.
# ``YieldStress`` / ``DP_cohesion`` / ``CapPressure`` / ``EpsQpShear``
# are named in ``_MATERIAL_BUCKET_TOKENS`` but their exact COMP_NAMES
# spelling is not documented anywhere, so they are left out here — no
# expectation, no check, rather than a guessed one that could
# false-positive. ``implexGuards`` / ``ladrunoBranch`` carry no
# ResponseType at all (every build writes ``C1..Cn``), so there is no
# real spelling to compare against and they are absent too.
_MATERIAL_BUCKET_EXPECTED_NAMES: "dict[str, tuple[str, ...]]" = {
    "pstress": ("p",),
    "j2stress": ("J2stress",),
    "volstrain": ("epsVol",),
    "j2strain": ("J2strain",),
    "backstress": tuple(f"BackStress_{i}" for i in range(1, 7)),
    "psi": ("psi",),
    "stateparameter": ("psi",),
    "yielddistance": ("yieldDistance",),
    "yieldfunction": ("yieldDistance",),
    "damage": ("d+", "d-"),
    "implexerror": ("implexError",),
    "avgimplexerror": ("avgImplexError",),
    "substeps": ("substeps_me", "substeps_capHit"),
    "substepsme": ("substeps_me", "substeps_capHit"),
    "ladrunosubsteps": ("substeps_me", "substeps_capHit"),
    "implexdetail": (
        "implexDetail_total", "implexDetail_dev", "implexDetail_vol",
        "implexDetail_clampFired", "implexDetail_clampCount",
        "implexDetail_f",
    ),
    "implexrefusals": (
        "implexRefusals_total", "implexRefusals_signChange",
        "implexRefusals_control", "implexRefusals_companion",
        "implexRefusals_commitLatched", "implexRefusals_latched",
    ),
}


def material_bucket_canonicals(token: str) -> "Optional[tuple[str, ...]]":
    """Canonical name per column for a ``material.<Token>`` bucket, else None.

    ``material.PStress`` → ``("material_mean_stress",)``;
    ``material.BackStress`` → the six ``back_stress_*`` names in the
    material's Voigt order. ``None`` for every other token — including
    ``material.stress`` / ``material.strain`` / ``material.pstrain`` /
    ``material.eqpstrain``, whose columns carry self-describing labels
    (``sigma11``, ``epsP11``, ``eqpstrain``) that
    :func:`continuum_canonical` already maps.
    """
    t = token.strip()
    if not t.lower().startswith(_MATERIAL_PREFIX):
        return None
    entry = _MATERIAL_BUCKET_TOKENS.get(t[len(_MATERIAL_PREFIX):].lower())
    if entry is None:
        return None
    return (entry,) if isinstance(entry, str) else entry


def material_bucket_expected_names(token: str) -> "Optional[tuple[str, ...]]":
    """The fork's documented real ``COMP_NAMES`` for ``material.<Token>``.

    ``None`` for a token not in :data:`_MATERIAL_BUCKET_EXPECTED_NAMES` —
    most tokens in :func:`material_bucket_canonicals` have no
    documented exact spelling to check against, and no expectation
    means no mismatch check.
    """
    t = token.strip()
    if not t.lower().startswith(_MATERIAL_PREFIX):
        return None
    return _MATERIAL_BUCKET_EXPECTED_NAMES.get(t[len(_MATERIAL_PREFIX):].lower())


def continuum_canonical(token: str) -> Optional[str]:
    """Map a continuum token to ``stress_ij`` / ``strain_ij`` /
    ``plastic_strain_ij`` (else None).

    Handles both the digit form (``sigma12``→``stress_xy``,
    ``eta11``/``eps11``→``strain_xx``, ``epsp11``→``plastic_strain_xx``)
    and the axis form (``sigma_xy``→``stress_xy``, ``eps_xx``→
    ``strain_xx``, ``gamma_xy``→``strain_xy``, ``epsp_xx``→
    ``plastic_strain_xx``). ``pstrain11``/``pstrain_xx`` (the
    ASDPlasticMaterial3D spelling) also map to ``plastic_strain_*``,
    and the scalar accumulated-PEEQ labels (``eqpstrain``, ``ebarP``,
    ``equivalentPlasticStrain``) map to ``equivalent_plastic_strain``.
    Matching is case-insensitive, so the out-of-plane ``sigma33`` /
    ``sigma_zz`` (however the recorder cases them) resolve to
    ``stress_zz``.
    """
    t = token.strip().lower()
    scalar = _CONTINUUM_SCALAR_TOKENS.get(t)
    if scalar is not None:
        return scalar
    m = _CONTINUUM_DIGIT_RE.match(t)
    if m is not None:
        root = _KIND_TO_ROOT[m.group("kind")]
        return f"{root}_{_AXIS[m.group('i')]}{_AXIS[m.group('j')]}"
    m = _CONTINUUM_AXIS_RE.match(t)
    if m is not None:
        root = _KIND_TO_ROOT[m.group("kind")]
        return f"{root}_{m.group('a')}{m.group('b')}"
    return None


def beam_canonical(token: str) -> Optional[tuple[str, Optional[int]]]:
    """``N_2`` → (``axial_force``, 2); ``N`` → (``axial_force``, None)."""
    m = _BEAM_RE.match(token.strip())
    if m is None:
        return None
    canonical = _BEAM_BASE_TO_CANONICAL.get(m.group("base"))
    if canonical is None:
        return None
    st = m.group("station")
    return canonical, (int(st) if st is not None else None)


# =====================================================================
# COLUMN_MAP block parsing
# =====================================================================

@dataclass(frozen=True)
class _Block:
    """One ``COLUMN_MAP`` row mapped to its slice of ``DATA``.

    Most blocks have ``multiplicity == 1`` (the block is exactly its
    ``comp_names``). Fiber buckets repeat a single column
    (``NUM_COMP==1``) once per fiber, so ``multiplicity`` == fibers and the
    block occupies ``len(comp_names) * multiplicity`` columns of ``DATA``.
    """
    level: int
    gauss_id: int
    comp_names: tuple[str, ...]
    col_start: int          # first DATA column for this block
    multiplicity: int = 1
    section_tag: int = -1
    fiber_id: int = -1
    #: ``comp_names`` are RESPONSE_CATALOG canonicals, not file labels
    #: (set only by :func:`_catalog_gauss_fit`). ``axial_force`` is no
    #: label any label map knows, so these names are used as they are.
    catalog_named: bool = False

    @property
    def width(self) -> int:
        return len(self.comp_names) * self.multiplicity

    def gp_column(self, gp: int, offset: int) -> int:
        """DATA column for component ``offset`` at Gauss point ``gp``.

        Layout is GP-major within the block: the component set repeats
        once per Gauss point.
        """
        return self.col_start + gp * len(self.comp_names) + offset

    @property
    def n_gauss(self) -> int:
        """Gauss points this block spans."""
        return self.multiplicity

    def gauss_index(self, gp: int) -> int:
        """Effective GP index: the repeat index when the block spans
        several Gauss points, else the block's own ``gauss_id`` (which is
        how ``section.force`` numbers its integration stations)."""
        return gp if self.multiplicity > 1 else self.gauss_id


def _decode_str(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def parse_blocks(bucket_grp: "h5py.Group") -> list[_Block]:
    """Parse a bucket's ``COLUMN_MAP`` + ``COMP_NAMES`` into blocks.

    Fails loud if the per-block ``COMP_NAMES`` line count or the summed
    block widths disagree with ``DATA`` — a malformed bucket should
    surface, not silently mis-slice.
    """
    cm = bucket_grp["COLUMN_MAP"]
    levels = np.asarray(cm["LEVELS"][...], dtype=np.int64).flatten()
    gauss = np.asarray(cm["GAUSS_ID"][...], dtype=np.int64).flatten()
    lines = _decode_str(cm.attrs["COMP_NAMES"]).split("\n")
    n_blocks = int(levels.size)
    if "MULTIPLICITY" in cm:
        mult = np.asarray(cm["MULTIPLICITY"][...], dtype=np.int64).flatten()
    else:
        mult = np.ones(n_blocks, dtype=np.int64)
    if "SECTION_TAG" in cm:
        sect = np.asarray(cm["SECTION_TAG"][...], dtype=np.int64).flatten()
    else:
        sect = np.full(n_blocks, -1, dtype=np.int64)
    if "FIBER_ID" in cm:
        fib = np.asarray(cm["FIBER_ID"][...], dtype=np.int64).flatten()
    else:
        fib = np.full(n_blocks, -1, dtype=np.int64)
    if len(lines) != n_blocks:
        raise ValueError(
            f"COLUMN_MAP has {n_blocks} rows but COMP_NAMES has "
            f"{len(lines)} lines — malformed .ladruno bucket."
        )
    blocks: list[_Block] = []
    col = 0
    for i in range(n_blocks):
        names = tuple(t.strip() for t in lines[i].split(",") if t.strip())
        m = int(mult[i]) if i < mult.size else 1
        blocks.append(_Block(
            level=int(levels[i]), gauss_id=int(gauss[i]),
            comp_names=names, col_start=col, multiplicity=m,
            section_tag=int(sect[i]) if i < sect.size else -1,
            fiber_id=int(fib[i]) if i < fib.size else -1,
        ))
        col += len(names) * m
    total = int(np.asarray(bucket_grp["DATA"].shape)[-1])
    if col != total:
        raise ValueError(
            f"COLUMN_MAP block widths sum to {col} but DATA has {total} "
            f"columns — malformed .ladruno bucket."
        )
    return blocks


def _class_prefix(bucket_key: str) -> str:
    """``12-Truss[1:0:0]`` → ``12-Truss`` (strip the bracket suffix)."""
    idx = bucket_key.find("[")
    return bucket_key[:idx] if idx >= 0 else bucket_key


def _gp_param_for(
    model_elements: "Optional[h5py.Group]", bucket_key: str,
) -> Optional[ndarray]:
    """Find ``QUADRATURE/GP_PARAM`` for the bucket's element class."""
    if model_elements is None:
        return None
    prefix = _class_prefix(bucket_key)
    for name in model_elements:
        if _class_prefix(name) != prefix:
            continue
        grp = model_elements[name]
        if "QUADRATURE" in grp and "GP_PARAM" in grp["QUADRATURE"]:
            return np.asarray(
                grp["QUADRATURE"]["GP_PARAM"][...], dtype=np.float64,
            )
    return None


# =====================================================================
# Generic ``C1..Cn`` columns — named from RESPONSE_CATALOG
# =====================================================================
#
# The module docstring's "the writer is self-describing" holds only as
# far as the ELEMENT is: the recorder writes the names the element hands
# it through ``output.tag("ResponseType", …)``. The fork's plain
# ``stress`` / ``strain`` responses on its plane elements emit no such
# tags (``LadrunoLST.cpp:872`` and siblings — only ``stressPlaneStrain``
# does), so the recorder falls back to ``C1,C2,…,Cn`` on a SINGLE
# element-level block. ``continuum_canonical("C1")`` is None, so every
# Gauss stress/strain column of every Ladruno plane element was dropped
# in silence (``available_components() == []``).
#
# For exactly those buckets the names come from ``RESPONSE_CATALOG``
# instead, keyed by the class name in the bucket key
# (``stress/33016-LadrunoLST[0:0:0]`` → ``LadrunoLST``) plus the
# ON_ELEMENTS token. The bracket's rule field is ``0``
# (NoIntegrationRule) for these buckets, so it cannot serve as the
# catalog's ``int_rule`` key — the BLOCK WIDTH picks the layout instead.
#
# Three more places on this seam would otherwise answer with an empty
# Gauss slab:
#
# * The ON_ELEMENTS token is the keyword the deck asked for, so the same
#   response lands under ``stress`` or ``stresses``. The catalog keys only
#   the first; the token is mapped through ``catalog_token_for_keyword``
#   (``stresses`` → ``stress``, ``axialForce`` → ``axial_force``) before
#   the lookup.
# * A Truss ``axialForce`` bucket names its one column (``N``) but writes
#   it as an ELEMENT-level block, which the Gauss level skips. The catalog
#   says this record is a one-Gauss-point ``axial_force``; a named
#   element-level block is expanded the same way when, and only when, its
#   own labels canonicalise to the catalog layout exactly.
# * These elements answer no ``integrationPoints`` response, so the file
#   has no ``QUADRATURE/GP_PARAM`` and the GPs would have no natural
#   coordinates (``(n, 0)``). A catalog-resolved bucket carries the
#   catalog's ``natural_coords``; the file's own GP_PARAM still wins.

_GENERIC_COMP_RE = re.compile(r"^C\d+$")


class GaussLayoutMismatch(Exception):
    """A generic ``C1..Cn`` block does not fit any catalog layout.

    Deliberately NOT a ``ValueError`` / ``KeyError``: the reads wrap
    ``parse_blocks`` in ``except (KeyError, ValueError): continue``, and
    a MIS-LABELLED component is worse than a missing one — this has to
    surface rather than be skipped.
    """


def _class_name(bucket_key: str) -> str:
    """``33016-LadrunoLST[0:0:0]`` → ``LadrunoLST``."""
    return _class_prefix(bucket_key).partition("-")[2]


def _catalog_token(token: str) -> str:
    """The ``RESPONSE_CATALOG`` token for an ON_ELEMENTS token.

    An OpenSees keyword the catalog aliases (``stresses``, ``axialForce``)
    maps to its catalog token; a token that already is one (``stress``,
    ``strain``, ``damage``) is returned as is.
    """
    mapped = catalog_token_for_keyword(token)
    return token if mapped is None else mapped


def _label_canonical(name: str) -> Optional[str]:
    """Canonical for a file column label: continuum first, then beam."""
    c = continuum_canonical(name)
    if c is not None:
        return c
    bc = beam_canonical(name)
    return None if bc is None else bc[0]


@dataclass(frozen=True)
class _CatalogFit:
    """An element-level bucket re-read as Gauss data from the catalog."""
    blocks: list[_Block]        # one block per Gauss point, canonical names
    natural_coords: ndarray     # (n_gp, d): the catalog layout's GP coords


def _catalog_gauss_fit(
    blocks: list[_Block], *, token: str, bucket_key: str,
) -> Optional[_CatalogFit]:
    """The catalog's Gauss reading of ``blocks``, or ``None`` to keep them.

    Only a single element-level block is a candidate, and only for a
    class the catalog knows under this token (``None`` otherwise: there
    is nothing to name it with). Generic ``C1..Cn`` columns are matched
    by width and raise :class:`GaussLayoutMismatch` when no single layout
    fits. Named columns are matched by their canonical labels and are
    kept as they are unless those labels ARE the catalog layout.
    """
    if len(blocks) != 1:
        return None
    b = blocks[0]
    if b.gauss_id >= 0 or not b.comp_names:
        return None
    class_name = _class_name(bucket_key)
    cat_token = _catalog_token(token)
    layouts = [
        lay for (cls, _rule, tok), lay in RESPONSE_CATALOG.items()
        if cls == class_name and tok == cat_token
    ]
    if not layouts:
        return None
    if all(_GENERIC_COMP_RE.match(n) for n in b.comp_names):
        fit = [
            lay for lay in layouts
            if lay.n_gauss_points * len(lay.component_layout) == b.width
        ]
        shapes = {(lay.n_gauss_points, lay.component_layout) for lay in fit}
        if len(shapes) != 1:
            offered = sorted({
                (lay.n_gauss_points,
                 lay.n_gauss_points * len(lay.component_layout))
                for lay in layouts
            })
            raise GaussLayoutMismatch(
                f"Bucket {token}/{bucket_key} has {b.width} unnamed "
                f"(C1..Cn) columns, but RESPONSE_CATALOG offers "
                f"{offered} (n_gauss_points, width) for class "
                f"{class_name!r} / token {cat_token!r} — "
                f"{'no' if not shapes else 'more than one'} layout "
                f"fits. Refusing to guess which component each column is: "
                f"a wrong component name is worse than a missing one. Fix "
                f"the RESPONSE_CATALOG entry, or have the element tag its "
                f"ResponseType names so the file names its own columns."
            )
    else:
        labels = tuple(
            _label_canonical(n) for n in b.comp_names
        ) * b.multiplicity
        fit = [
            lay for lay in layouts
            if lay.component_layout * lay.n_gauss_points == labels
        ]
        if not fit:
            return None
    coords = {
        np.asarray(lay.natural_coords, dtype=np.float64).tobytes():
            np.asarray(lay.natural_coords, dtype=np.float64)
        for lay in fit
    }
    if len(coords) != 1:
        raise GaussLayoutMismatch(
            f"Bucket {token}/{bucket_key}: RESPONSE_CATALOG has layouts "
            f"of class {class_name!r} / token {cat_token!r} that fit its "
            f"{b.width} columns but place the Gauss points differently. "
            f"Refusing to guess where each point is."
        )
    n_gp, names = fit[0].n_gauss_points, fit[0].component_layout
    return _CatalogFit(
        blocks=[
            _Block(
                level=1, gauss_id=gp, comp_names=names,
                col_start=b.col_start + gp * len(names),
                catalog_named=True,
            )
            for gp in range(n_gp)
        ],
        natural_coords=next(iter(coords.values())),
    )


def resolve_generic_gauss_blocks(
    blocks: list[_Block], *, token: str, bucket_key: str,
) -> list[_Block]:
    """``blocks`` with a catalog-known element-level block resolved.

    Returns ``blocks`` itself (the same object) unless
    :func:`_catalog_gauss_fit` re-reads the bucket as Gauss data; then it
    returns one block per Gauss point carrying the catalog's (already
    canonical) component names. A class the catalog knows for this token
    whose generic ``C1..Cn`` width fits no layout raises
    :class:`GaussLayoutMismatch`.
    """
    fit = _catalog_gauss_fit(blocks, token=token, bucket_key=bucket_key)
    return blocks if fit is None else fit.blocks


def _select_rows(
    bucket_grp: "h5py.Group", element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray]]":
    """Return ``(row_indices, element_ids)`` after the id filter, or None."""
    all_ids = np.asarray(bucket_grp["ID"][...], dtype=np.int64).flatten()
    if element_ids is None:
        return np.arange(all_ids.size, dtype=np.int64), all_ids
    want = np.asarray(element_ids, dtype=np.int64)
    rows = np.where(np.isin(all_ids, want))[0]
    if rows.size == 0:
        return None
    return rows, all_ids[rows]


# =====================================================================
# Gauss reads (continuum stress / strain)
# =====================================================================


class GaussColumnDroppedWarning(UserWarning):
    """A material-level column was discarded — nothing could name it.

    Raised (once per bucket, per read) by :func:`gauss_available` for a
    ``material.<Token>`` bucket whose token is not in
    :data:`_MATERIAL_BUCKET_TOKENS` and whose labels
    :func:`continuum_canonical` does not know either. The column is
    dropped: it never reaches ``available_components()`` and cannot be
    plotted. A silent drop is indistinguishable from a material that
    never wrote the quantity — the ASDPlasticMaterial3D material-level
    buckets went missing exactly that way (ADR 0105 Amendment 1), so say
    so.

    Scoped to ``material.`` buckets on purpose. Elsewhere a label the
    Gauss level cannot name is routinely another level's business — a
    ``section.force`` station carries ``P``/``Mz``, which
    :func:`section_canonical` owns — so warning there would be noise,
    not news.
    """


class GaussColumnNameMismatchWarning(UserWarning):
    """A material bucket's real COMP_NAMES disagree with the fork's order.

    Raised (once per bucket, per read) by :func:`gauss_available` when a
    ``material.<Token>`` bucket writes real (non-generic) column names
    that are in :data:`_MATERIAL_BUCKET_EXPECTED_NAMES` but do not match
    what the file actually wrote. :data:`_MATERIAL_BUCKET_TOKENS`
    resolves those columns BY POSITION regardless — the table wins
    whenever the slot count matches, so a reordered or renamed column
    would otherwise be mislabelled in silence rather than merely
    dropped. This warning does not change what gets returned (reordering
    would be its own guess); it only says the file and the table
    disagree.
    """


def _block_canonicals(token: str, b: _Block) -> list[Optional[str]]:
    """Canonical (or None) per column of ``b``, by token then by label.

    A ``material.<Token>`` bucket in :data:`_MATERIAL_BUCKET_TOKENS` is
    resolved by COLUMN POSITION (its labels are material-private and
    collide across levels — ``p`` vs the section force ``P``). Everything
    else, including a material bucket whose column count does not match
    the table, falls back to the label map. A catalog-resolved block
    already carries canonicals.
    """
    if b.catalog_named:
        return list(b.comp_names)
    names = material_bucket_canonicals(token)
    if names is not None and len(names) == len(b.comp_names):
        return list(names)
    return [continuum_canonical(name) for name in b.comp_names]


def _name_mismatch(
    token: str, b: _Block,
) -> "Optional[tuple[tuple[str, ...], tuple[str, ...]]]":
    """``(actual, expected)`` if ``b``'s real COMP_NAMES disagree with the
    fork's documented order for ``token`` — else ``None``.

    Only checked when the by-position table applies (the block's width
    matches the table, the same condition :func:`_block_canonicals` uses
    to resolve by position) and the names are not the generic ``C1..Cn``
    fallback: a generic bucket has nothing to compare, and a width
    mismatch already falls back to the label map.
    """
    if not b.comp_names or all(_GENERIC_COMP_RE.match(n) for n in b.comp_names):
        return None
    canon = material_bucket_canonicals(token)
    if canon is None or len(canon) != len(b.comp_names):
        return None
    expected = material_bucket_expected_names(token)
    if expected is None or len(expected) != len(b.comp_names):
        return None
    actual = tuple(b.comp_names)
    if actual == expected:
        return None
    return actual, expected


def gauss_available(on_elements: "h5py.Group") -> set[str]:
    out: set[str] = set()
    dropped: dict[str, set[str]] = {}
    mismatched: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for token in on_elements:
        if is_fiber_token(token):
            continue
        for key in on_elements[token]:
            try:
                blocks = parse_blocks(on_elements[token][key])
            except (KeyError, ValueError):
                continue
            blocks = resolve_generic_gauss_blocks(
                blocks, token=token, bucket_key=key,
            )
            bucket = f"{token}/{key}"
            for b in blocks:
                # Element-level rows are not Gauss data. Fiber buckets are
                # excluded by token before this loop.
                if b.gauss_id < 0:
                    continue
                if bucket not in mismatched:
                    mm = _name_mismatch(token, b)
                    if mm is not None:
                        mismatched[bucket] = mm
                for name, c in zip(b.comp_names, _block_canonicals(token, b)):
                    if c is not None:
                        out.add(c)
                    elif token.lower().startswith(_MATERIAL_PREFIX):
                        dropped.setdefault(bucket, set()).add(name)
    for bucket, labels in dropped.items():
        warnings.warn(
            f"Gauss column(s) {sorted(labels)} of bucket {bucket!r} were "
            f"dropped: apeGmsh has no canonical component for that "
            f"material bucket, so its columns are not listed by "
            f"available_components() and cannot be read or plotted. "
            f"Element class {_class_name(bucket.split('/', 1)[-1])!r}. "
            f"Add the bucket token to the .ladruno reader's material "
            f"bucket table if the quantity should be readable.",
            GaussColumnDroppedWarning,
            stacklevel=2,
        )
    for bucket, (actual, expected) in mismatched.items():
        warnings.warn(
            f"Bucket {bucket!r} COMP_NAMES {list(actual)} disagree with "
            f"the fork's documented order {list(expected)} for this "
            f"material response. apeGmsh still resolves its columns by "
            f"POSITION (the table's slot order), so a reordered or "
            f"renamed file column is read under the WRONG canonical "
            f"name. Element class {_class_name(bucket.split('/', 1)[-1])!r}.",
            GaussColumnNameMismatchWarning,
            stacklevel=2,
        )
    return out


def read_gauss_slab(
    on_elements: "h5py.Group",
    model_elements: "Optional[h5py.Group]",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumGP], element_index, natural_coords[sumGP, d])``.

    Stitches every bucket whose blocks expose ``component`` at a Gauss
    point. ``None`` if nothing matches. Two tokens can cover the same
    (element, GP) with the same component — see
    :func:`_dedupe_gauss_columns`, which resolves that overlap.
    """
    values_parts: list[ndarray] = []
    eidx_parts: list[ndarray] = []
    coord_parts: list[ndarray] = []
    gid_parts: list[ndarray] = []
    named_parts: list[ndarray] = []

    for token in on_elements:
        if is_fiber_token(token):
            continue
        token_grp = on_elements[token]
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            fit = _catalog_gauss_fit(blocks, token=token, bucket_key=key)
            # Whether the file's own blocks are read as they are is the
            # provenance flag the overlap tie-break needs.
            from_file_names = fit is None
            if fit is not None:
                blocks = fit.blocks
            gp_blocks = [b for b in blocks if b.gauss_id >= 0]
            if not gp_blocks:
                continue
            # (block, col-within-DATA) pairs that hold the component.
            # (block, gp, col) triples that hold the component. A block
            # spanning several Gauss points contributes one hit per GP.
            hits: list[tuple[_Block, int, int]] = []
            for b in gp_blocks:
                for off, c in enumerate(_block_canonicals(token, b)):
                    if c != component:
                        continue
                    for gp in range(b.n_gauss):
                        hits.append((b, gp, b.gp_column(gp, off)))
            if not hits:
                continue
            sel = _select_rows(bucket, element_ids)
            if sel is None:
                continue
            rows, sel_ids = sel
            # Hyperslab: only the requested steps, rows and hit columns.
            hit_cols = np.array(sorted({col for _, _, col in hits}))
            col_pos = {int(c): i for i, c in enumerate(hit_cols)}
            data = read_hyperslab(bucket["DATA"], t_idx, rows, hit_cols)
            # The file's own GP coordinates win. An element answering no
            # ``integrationPoints`` leaves none, and then a catalog-read
            # bucket places its points by the catalog's rule.
            gp_param = _gp_param_for(model_elements, key)
            if gp_param is None and fit is not None:
                gp_param = fit.natural_coords

            # One slab column per (element, matching GP). GP-major within
            # an element so natural_coords line up with element_index.
            for b, gp, col in hits:
                vals = data[:, :, col_pos[col]]               # (T, E_sel)
                values_parts.append(vals)
                eidx_parts.append(sel_ids)
                g_idx = b.gauss_index(gp)
                gid_parts.append(np.full(sel_ids.size, g_idx, dtype=np.int64))
                named_parts.append(
                    np.full(sel_ids.size, from_file_names, dtype=bool)
                )
                if gp_param is not None and g_idx < gp_param.shape[0]:
                    nat = np.tile(gp_param[g_idx], (sel_ids.size, 1))
                else:
                    nat = np.zeros((sel_ids.size, 0), dtype=np.float64)
                coord_parts.append(nat)

    if not values_parts:
        return None
    values = np.concatenate(values_parts, axis=1)
    element_index = np.concatenate(eidx_parts)
    max_dim = max((c.shape[1] for c in coord_parts), default=0)
    coords = (
        np.concatenate(
            [c if c.shape[1] == max_dim
             else np.full((c.shape[0], max_dim), np.nan) for c in coord_parts],
            axis=0,
        )
        if max_dim else np.zeros((element_index.size, 0), dtype=np.float64)
    )
    keep = _dedupe_gauss_columns(
        values, element_index,
        np.concatenate(gid_parts), np.concatenate(named_parts),
        component=component,
    )
    if keep is None:
        return values, element_index, coords
    return values[:, keep], element_index[keep], coords[keep]


def _dedupe_gauss_columns(
    values: ndarray, element_index: ndarray, gauss_index: ndarray,
    named: ndarray, *, component: str,
) -> "Optional[ndarray]":
    """Column indices to keep, or ``None`` when nothing overlaps.

    One ``(element, Gauss point)`` slot can be covered by TWO tokens: the
    fork emits the same plane stress under the unnamed ``stress`` (named
    here from ``RESPONSE_CATALOG``) and under ``stressesPlaneStrain``,
    which tags its own ``sigma11…sigma33`` and carries the out-of-plane
    component as well. Concatenating both double-counts every in-plane
    column, and the slab then no longer lines up with the components that
    only ONE token carries (``stress_zz``) — which is how a derived
    scalar like ``von_mises_stress`` ends up broadcasting (T, 204) into
    (T, 408).

    Precedence: the bucket whose COMP_NAMES the FILE wrote wins over a
    catalog-reconstructed one. The named bucket is self-describing and
    may be a superset, so preferring it keeps every slot on one
    consistent source. Duplicates must agree numerically — they are the
    same material state read twice — so a genuine disagreement means a
    mis-labelled column and raises rather than picking one arbitrarily.
    """
    keys = element_index.astype(np.int64) * (int(gauss_index.max()) + 1)
    keys = keys + gauss_index
    _uniq, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True,
    )
    if counts.max() <= 1:
        return None
    # Sort by slot, then file-named first; the leader of each run wins.
    order = np.lexsort((~named, keys))
    first = np.unique(keys[order], return_index=True)[1]
    winner = order[first]                       # one column per unique slot
    dropped = np.setdiff1d(
        np.arange(keys.size, dtype=np.int64), winner, assume_unique=False,
    )
    kept_for_dropped = winner[inverse[dropped]]
    if not np.allclose(
        values[:, dropped], values[:, kept_for_dropped],
        rtol=1e-9, atol=0.0, equal_nan=True,
    ):
        bad = int(np.argmax(np.any(
            ~np.isclose(
                values[:, dropped], values[:, kept_for_dropped],
                rtol=1e-9, atol=0.0, equal_nan=True,
            ), axis=0,
        )))
        raise GaussLayoutMismatch(
            f"Two .ladruno buckets report different values for "
            f"{component!r} at element {int(element_index[dropped[bad]])} "
            f"Gauss point {int(gauss_index[dropped[bad]])}. Same material "
            f"state read twice must agree, so one of the buckets' columns "
            f"is mis-labelled — refusing to pick one arbitrarily. Check "
            f"the element's ResponseType tags against its RESPONSE_CATALOG "
            f"component_layout."
        )
    return np.sort(winner)      # keep the original column order


# =====================================================================
# Line-station reads (beam internal-force diagrams)
# =====================================================================

def line_station_available(on_elements: "h5py.Group") -> set[str]:
    """Line-station canonicals the file's buckets can answer.

    A ``localForce`` bucket that :func:`_local_force_2d_gate` refuses
    lists nothing and warns once, naming its class: its columns name no
    component reliably, and a read of its elements raises the same reason.
    A ``localForce`` bucket whose ``COLUMN_MAP`` does not parse (its names
    do not cover its DATA columns) lists nothing and warns, naming it.
    A ``basicForce`` component that :func:`_basic_force_refusal` refuses
    is left out with a warning naming the bucket and the component.
    """
    out: set[str] = set()
    for token in on_elements:
        for key in on_elements[token]:
            bucket = on_elements[token][key]
            element_stations: dict[str, list[int]] = {}
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError) as exc:
                if token == "localForce":
                    warnings.warn(
                        f"localForce bucket {key!r} (element class "
                        f"{_class_name(key) or '<unparsed>'!r}) is "
                        f"unreadable: {exc!s} Its line stations are left "
                        f"out.",
                        UserWarning, stacklevel=2,
                    )
                continue
            try:
                alias_2d = _local_force_2d_gate(bucket, blocks, token)
            except ValueError as exc:
                warnings.warn(
                    f"{exc} Its line stations are left out.",
                    UserWarning, stacklevel=2,
                )
                continue
            for b in blocks:
                for name in b.comp_names:
                    if b.level == 2:
                        c = section_canonical(name)
                        if c is not None:
                            out.add(c)
                    else:
                        bc = _station_canonical(name, token, alias_2d)
                        if bc is not None:
                            element_stations.setdefault(bc[0], []).append(
                                bc[1] if bc[1] is not None else 1)
            for comp, stations in element_stations.items():
                if token == "basicForce":
                    reason = _basic_force_refusal(key, comp, sorted(stations))
                    if reason is not None:
                        warnings.warn(reason, UserWarning, stacklevel=2)
                        continue
                out.add(comp)
    return out


#: A 2-D ``localForce`` bucket names its shear and moment ``V``/``M``
#: (``N_1,V_1,M_1,N_2,V_2,M_2``); a 2-D beam has one of each, ``Vy``/``Mz``.
#: Only for ``localForce``: ``basicForce``'s ``M_1``/``M_2`` are basic-system
#: end moments, not section forces. It applies only to a bucket that
#: :func:`_local_force_2d_gate` admits.
_LOCAL_FORCE_2D_ALIAS = {"V": "Vy", "M": "Mz"}

#: A ``localForce`` station token: ``N_1`` and ``N1`` both occur
#: (DispBeamColumn2d and LadrunoDispBeamColumn2d drop the underscore).
_LOCAL_FORCE_TOKEN_RE = re.compile(r"^(?P<base>[A-Za-z]+)_?(?P<station>\d+)$")

#: The 3-D shear, torsion and moment bases a 2-D spelling must not mix with.
_LOCAL_FORCE_3D_BASES = frozenset({"Vy", "Vz", "T", "My", "Mz"})

#: ``basicForce``'s two-station components: the basic-system end moments
#: ``Mz_1,Mz_2`` and ``My_1,My_2`` of a 3-D ``N,Mz_1,Mz_2,My_1,My_2,T``
#: bucket (#2011). Every fork class that names them (ElasticBeam3d,
#: ModElasticBeam3d, Pipe, ForceBeamColumn3d, ForceBeamColumn3dThermal,
#: MixedBeamColumn3d, MixedBeamColumnAsym3d at fork c744d6e70) reports its
#: ``localForce`` end moments as these same values, member loads included
#: (``Mz_i = q1``, ``Mz_j = q2``, ``My_i = q3``, ``My_j = q4``), so they take
#: the ``localForce`` station signs unchanged: ``[-M_1, +M_2]``.
_BASIC_FORCE_END_MOMENTS = frozenset({"bending_moment_y", "bending_moment_z"})


def _basic_force_refusal(
    key: str, component: str, stations: list[int],
) -> Optional[str]:
    """Why a ``basicForce`` component reads as no line station, or ``None``.

    A single-station component (``N``, ``T``) and an end-moment pair at
    stations ``[1, 2]`` read. Any other multi-station component is refused:
    no basic-to-section identity is known for it.
    """
    if len(stations) == 1:
        return None
    if component in _BASIC_FORCE_END_MOMENTS and stations == [1, 2]:
        return None
    return (
        f"basicForce bucket {key!r} (element class "
        f"{_class_name(key) or '<unparsed>'!r}) names {component!r} at "
        f"stations {stations}; only the end moments "
        f"{sorted(_BASIC_FORCE_END_MOMENTS)} at stations [1, 2] map to the "
        f"section convention. Its {component!r} line stations are left out."
    )


def _file_spatial_dim(group: "h5py.Group") -> Optional[int]:
    """The ``.ladruno`` file's spatial dimension, or ``None`` if unknown.

    ``INFO/SPATIAL_DIM`` when the file carries it, else the one width of
    1, 2 or 3 that every stage's ``MODEL/NODES/COORDINATES`` agrees on
    (the reader's own rule, #2081).
    """
    f = group.file
    if "INFO" in f and "SPATIAL_DIM" in f["INFO"].attrs:
        raw = np.asarray(f["INFO"].attrs["SPATIAL_DIM"])
        if raw.size:
            return int(raw.flat[0])
    widths: set[int] = set()
    for name in f:
        if not name.startswith("MODEL_STAGE["):
            continue
        stage = f[name]
        if ("MODEL" in stage and "NODES" in stage["MODEL"]
                and "COORDINATES" in stage["MODEL"]["NODES"]):
            coords = stage["MODEL"]["NODES"]["COORDINATES"]
            if len(coords.shape) == 2:
                widths.add(int(coords.shape[1]))
    if len(widths) == 1 and widths <= {1, 2, 3}:
        return widths.pop()
    return None


def _local_force_2d_gate(
    bucket: "h5py.Group", blocks: list[_Block], token: str,
) -> bool:
    """True when a ``localForce`` bucket's 2-D names read as ``Vy``/``Mz``.

    False for a bucket that spells no 2-D ``V``/``M`` name (or another
    token): its names read as they are. A bucket that does spell them is
    admitted only when its ``DATA`` holds exactly the tagged columns, one
    per name, it spells no 3-D base beside them, the file is 2-D, and its
    class is no ``...3d`` class. Every other such bucket raises
    :class:`ValueError` naming the bucket and its class, the gate #2010 put
    on MPCO.

    ComponentElement3d and ForceBeamColumnCBDI3d tag a 12-value
    ``localForce`` with the six 2-D names. The fork's recorder keeps the
    declared 6 columns and fills the row (NaN at the pin, zeros before
    WP-163), so the bucket's width agrees with its names and only the
    dimension and class gates can tell. A 12-wide bucket under 6 names is
    refused by the width gate.
    """
    if token != "localForce":
        return False
    names = [n.strip() for b in blocks for n in b.comp_names]
    bases = {
        m.group("base") for m in map(_LOCAL_FORCE_TOKEN_RE.match, names)
        if m is not None
    }
    spelled_2d = bases & set(_LOCAL_FORCE_2D_ALIAS)
    if not spelled_2d:
        return False
    key = bucket.name.rsplit("/", 1)[-1]
    cls = _class_name(key) or "<unparsed>"
    where = f"localForce bucket {key!r} (element class {cls!r})"
    width = int(np.asarray(bucket["DATA"].shape)[-1])
    if any(b.multiplicity != 1 for b in blocks) or len(names) != width:
        raise ValueError(
            f"{where}: its DATA is {width} columns wide per element but it "
            f"tags {len(names)} components {names}; the 2-D names cannot "
            f"be matched to the columns."
        )
    spelled_3d = bases & _LOCAL_FORCE_3D_BASES
    if spelled_3d:
        raise ValueError(
            f"{where}: mixes the 2-D names {sorted(spelled_2d)} with the "
            f"3-D {sorted(spelled_3d)}; the shear and moment axes are "
            f"ambiguous."
        )
    ndm = _file_spatial_dim(bucket)
    if ndm != 2:
        raise ValueError(
            f"{where}: tags its columns with the 2-D names {names}, but the "
            f"file's spatial dimension is {ndm!r}, not 2; the shear and "
            f"moment axes cannot be read from them."
        )
    if not _class_name(key) or cls.lower().endswith("3d"):
        raise ValueError(
            f"{where}: the class records its localForce under the 2-D "
            f"names {names}; a 3-D class (or an unparsed key) cannot be "
            f"read as a 2-D beam."
        )
    return True


def _section_served_ids(on_elements: "h5py.Group", component: str) -> ndarray:
    """Element ids a section-level bucket serves ``component`` for.

    The rank-0 source of :func:`read_line_station_slab`: a bucket with a
    ``LEVELS==2`` block naming ``component`` gives every one of its
    elements section stations, which win over ``localForce``.
    """
    parts: list[ndarray] = []
    for token in on_elements:
        for key in on_elements[token]:
            bucket = on_elements[token][key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            if any(b.level == 2 and section_canonical(n) == component
                   for b in blocks for n in b.comp_names):
                parts.append(np.asarray(
                    bucket["ID"][...], dtype=np.int64).flatten())
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _station_canonical(
    name: str, token: str, alias_2d: bool,
) -> Optional[tuple[str, Optional[int]]]:
    """:func:`beam_canonical`, with the gated 2-D ``localForce`` spellings.

    ``alias_2d`` is :func:`_local_force_2d_gate`'s verdict on the bucket:
    when True, ``V``/``M`` read as ``Vy``/``Mz`` and ``N1`` as ``N_1``.
    """
    if token == "localForce" and alias_2d:
        m = _LOCAL_FORCE_TOKEN_RE.match(name.strip())
        if m is not None:
            base = m.group("base")
            name = (f"{_LOCAL_FORCE_2D_ALIAS.get(base, base)}_"
                    f"{m.group('station')}")
    return beam_canonical(name)


def read_line_station_slab(
    on_elements: "h5py.Group",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
    model_elements: "Optional[h5py.Group]" = None,
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumS], element_index, station_natural_coord)``.

    Two bucket flavours feed the beam line diagram:

    * **element-level** (``localForce`` / ``basicForce``, ``LEVELS==0``) —
      one block of named columns. ``localForce`` end forces are signed per
      component by ``local_force_station_signs`` into the force-beam
      ``section force`` convention, as on every other route, so the
      diagram is continuous across element boundaries; single-station
      ``basicForce`` is left as-is. Stations are evenly
      spaced over ``[-1, +1]``.
    * **section-level** (``section.force`` / ``section.deformation``,
      ``LEVELS==2``) — one block per integration station. The station's
      natural coordinate is read from the element's ``QUADRATURE/GP_PARAM``
      keyed by ``GAUSS_ID`` (force-based beams), not synthesized.

    An element takes its stations from its first source only, in this
    order: section-level, then ``localForce``, then any other element-level
    bucket (a truss ``axialForce``). A force beam that also records
    ``localForce`` or ``basicForce`` keeps its section stations, and an
    elastic beam (no section response) gets its two ``localForce``
    stations, as MPCO's reader does.
    """
    found: list[tuple[int, tuple[ndarray, ndarray, ndarray]]] = []
    for token in on_elements:
        token_grp = on_elements[token]
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            if any(b.level == 2 for b in blocks):
                rank = 0
                res = _read_section_stations(
                    bucket, blocks, component, key, model_elements,
                    t_idx=t_idx, element_ids=element_ids,
                )
            else:
                rank = 1 if token == "localForce" else 2
                res = _read_element_stations(
                    bucket, blocks, component, token,
                    t_idx=t_idx, element_ids=element_ids,
                )
            if res is not None:
                found.append((rank, res))

    values_parts: list[ndarray] = []
    eidx_parts: list[ndarray] = []
    station_parts: list[ndarray] = []
    covered: set[int] = set()
    for rank in (0, 1, 2):
        served: set[int] = set()
        for r, (v, ei, st) in found:
            keep = ~np.isin(ei, list(covered))
            if r != rank or not keep.any():
                continue
            values_parts.append(v[:, keep])
            eidx_parts.append(ei[keep])
            station_parts.append(st[keep])
            served.update(int(e) for e in ei[keep])
        covered |= served

    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(eidx_parts),
        np.concatenate(station_parts),
    )


def _read_element_stations(
    bucket: "h5py.Group", blocks: list[_Block], component: str, token: str,
    *, t_idx: ndarray, element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Element-level (``localForce``/``basicForce``) line stations.

    A ``localForce`` bucket that :func:`_local_force_2d_gate` refuses
    raises its reason, whatever the component, when the read asks for one
    of its elements (``element_ids=None`` asks for all) that no
    section-level bucket serves with this component: its columns name
    none reliably. An element with section stations keeps them, since the
    section source ranks first (:func:`read_line_station_slab`), and
    reads of other elements pass the bucket over.

    A ``basicForce`` end-moment pair (``Mz_1,Mz_2`` / ``My_1,My_2``) takes
    the ``localForce`` station signs, so it reads ``[-M_1, +M_2]``, the
    stations ``localForce`` gives the same element (#2011). A component
    :func:`_basic_force_refusal` refuses reads as no stations and warns,
    naming the bucket and the component.
    """
    try:
        alias_2d = _local_force_2d_gate(bucket, blocks, token)
    except ValueError as exc:
        hit = _select_rows(bucket, element_ids)
        if hit is not None:
            served = _section_served_ids(bucket.parent.parent, component)
            bare = hit[1][~np.isin(hit[1], served)]
            if bare.size:
                raise ValueError(
                    f"{exc} (requested element(s) {bare[:10].tolist()} are "
                    f"in this bucket, with no section stations for "
                    f"{component!r})"
                ) from exc
        return None
    # station (1-based, None→1) → DATA column for this component.
    station_to_col: dict[int, int] = {}
    for b in blocks:
        for off, name in enumerate(b.comp_names):
            bc = _station_canonical(name, token, alias_2d)
            if bc is None or bc[0] != component:
                continue
            station = bc[1] if bc[1] is not None else 1
            station_to_col[station] = b.col_start + off
    if not station_to_col:
        return None
    sel = _select_rows(bucket, element_ids)
    if sel is None:
        return None
    rows, sel_ids = sel
    stations = sorted(station_to_col)
    n_st = len(stations)
    if token == "basicForce":
        key = bucket.name.rsplit("/", 1)[-1]
        reason = _basic_force_refusal(key, component, stations)
        if reason is not None:
            warnings.warn(reason, UserWarning, stacklevel=2)
            return None
    T = int(np.size(t_idx))
    E = sel_ids.size
    # (T, E, n_st), read as a hyperslab of the station columns only.
    out = read_hyperslab(
        bucket["DATA"], t_idx, rows,
        np.array([station_to_col[st] for st in stations]),
    )
    # localForce end forces → the force-beam ``section force`` signs,
    # the one table every route applies. basicForce's end-moment pair is
    # the localForce end moments (:data:`_BASIC_FORCE_END_MOMENTS`).
    if token in ("localForce", "basicForce") and n_st == 2:
        out *= local_force_station_signs((component,))[:, 0]
    return (
        out.reshape(T, E * n_st),
        np.repeat(sel_ids, n_st),
        np.tile(_station_xi(n_st), E),
    )


def _read_section_stations(
    bucket: "h5py.Group", blocks: list[_Block], component: str,
    bucket_key: str, model_elements: "Optional[h5py.Group]",
    *, t_idx: ndarray, element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Section-level (``section.force``/``section.deformation``) stations.

    Each ``LEVELS==2`` block is one integration station; the station's
    natural coordinate comes from the element's ``GP_PARAM`` keyed by
    ``GAUSS_ID``.
    """
    # gauss_id → DATA column carrying this component.
    gp_to_col: dict[int, int] = {}
    for b in blocks:
        if b.level != 2:
            continue
        for off, name in enumerate(b.comp_names):
            if section_canonical(name) == component:
                gp_to_col[b.gauss_id] = b.col_start + off
    if not gp_to_col:
        return None
    sel = _select_rows(bucket, element_ids)
    if sel is None:
        return None
    rows, sel_ids = sel
    gp_param = _gp_param_for(model_elements, bucket_key)

    gauss_ids = sorted(gp_to_col)
    n_st = len(gauss_ids)
    T = int(np.size(t_idx))
    E = sel_ids.size
    # (T, E, n_st), read as a hyperslab of the station columns only.
    out = read_hyperslab(
        bucket["DATA"], t_idx, rows,
        np.array([gp_to_col[gid] for gid in gauss_ids]),
    )
    xi = np.empty(n_st, dtype=np.float64)
    fallback = _station_xi(n_st)
    for s_i, gid in enumerate(gauss_ids):
        if gp_param is not None and 0 <= gid < gp_param.shape[0]:
            xi[s_i] = float(gp_param[gid, 0])
        else:
            xi[s_i] = fallback[s_i]
    return (
        out.reshape(T, E * n_st),
        np.repeat(sel_ids, n_st),
        np.tile(xi, E),
    )


def _station_xi(n_stations: int) -> ndarray:
    if n_stations == 1:
        return np.array([0.0])
    return np.linspace(-1.0, 1.0, n_stations)


# =====================================================================
# Element reads (token-driven raw blocks)
# =====================================================================
#
# Design (aligned with the parallel effort): element reads are
# **token-driven**, not neutral-canonical. The component IS the file's
# ``ON_ELEMENTS/<token>`` key (``basicForce`` / ``localForce`` /
# ``globalForce`` / ``force``) and the slab carries the element's full raw
# output vector ``(T, E, NUM_COLUMNS)`` in the file's ``COMP_NAMES`` column
# order. The raw token reads stay exactly that. One neutral view sits on
# top of them: ``nodal_resisting_{force,moment}_{x,y,z}`` decoded
# from the ``globalForce`` token (see "Canonical nodal resisting forces"
# below). The other neutral views live on the *other* levels —
# ``line_stations`` (``axial_force``…) and ``gauss`` (``stress_xx``…).
#
# Three details pinned here (flag if the sibling branch chose otherwise):
#   1. npe = raw ``NUM_COLUMNS`` (localForce→12, quad force→8, basicForce→1).
#   2. ``available_components(ELEMENTS)`` lists token keys whose blocks are
#      all element-level (``LEVELS==0``) — excludes Gauss stress/strain.
#   3. ``ElementSlab`` is unextended; column meaning follows the file's
#      ``COMP_NAMES`` (not carried on the slab).


def element_available(
    on_elements: "h5py.Group", *, ndm: int,
) -> set[str]:
    """ON_ELEMENTS token keys that are element-level (all blocks LEVELS==0),
    plus the canonical nodal resisting forces the ``globalForce`` buckets
    decode to (``ndm`` is the file's spatial dimension).

    Token-driven: the token *is* the component. Gauss-level tokens
    (stress/strain, LEVELS==4) are excluded — they belong to read_gauss.
    """
    out: set[str] = set(_nodal_force_available(on_elements, ndm=ndm))
    for token in on_elements:
        token_grp = on_elements[token]
        saw_bucket = False
        all_element_level = True
        for key in token_grp:
            try:
                blocks = parse_blocks(token_grp[key])
            except (KeyError, ValueError):
                continue
            saw_bucket = True
            if any(b.level != 0 for b in blocks):
                all_element_level = False
                break
        if saw_bucket and all_element_level:
            out.add(token)
    return out


class ElementWidthMismatchWarning(UserWarning):
    """Buckets of a different column width were dropped from an element read.

    Raised (once per read) by :func:`read_element_slab` when one response
    token spans buckets whose component counts differ. They cannot share
    one dense ``(T, E, ncol)`` slab, so the first width wins and the rest
    are skipped — a deliberate choice, but a silent one until now: the
    caller got fewer elements than it asked for with nothing to
    distinguish that from elements which never recorded the quantity
    (the same indistinguishability :class:`GaussColumnDroppedWarning`
    exists for).

    The mixed case is real, not hypothetical. The fork sizes a
    ``zeroLength`` ``force`` response by ``ndf1 + ndf2``, so a 3-D model
    pairing ordinary (3,3) nodes with u-p (3,4) nodes writes 6- and
    7-column buckets under one ``force`` token.
    """


def read_element_slab(
    on_elements: "h5py.Group",
    token: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
    ndm: "Optional[int]" = None,
) -> "Optional[tuple[ndarray, ndarray]]":
    """Return ``(values[T, E, ncol], element_ids)`` for ``ON_ELEMENTS/<token>``.

    The raw element output block, sliced to the requested time steps and
    elements, in the file's ``COMP_NAMES`` column order. Buckets of different
    column width under one token (e.g. 2-D vs 3-D beams sharing a
    ``localForce`` token) can't share one ``(T, E, ncol)`` slab — the first
    width wins and mismatched buckets are skipped (homogeneous models, the
    common case, always match). A skip is never silent: it raises
    :class:`ElementWidthMismatchWarning` naming both widths and the
    elements left out.

    A ``nodal_resisting_*`` canonical name is the one exception to the
    token rule: it reads the ``globalForce`` token as ``(T, E, n_nodes)``,
    the MPCO reader's layout (see :func:`read_nodal_force_slab`). It needs
    ``ndm``, the file's spatial dimension.
    """
    if token in PER_ELEMENT_NODAL_FORCES:
        if ndm is None:
            raise TypeError(
                f"read_element_slab({token!r}) decodes a canonical nodal "
                f"resisting force and needs ndm= (the spatial dimension)."
            )
        return read_nodal_force_slab(
            on_elements, token, ndm=ndm, t_idx=t_idx, element_ids=element_ids,
        )
    if token not in on_elements:
        return None
    token_grp = on_elements[token]
    values_parts: list[ndarray] = []
    eid_parts: list[ndarray] = []
    ncol_ref: "Optional[int]" = None
    ncol_key: "Optional[str]" = None
    dropped: list[tuple[str, int, ndarray]] = []

    for key in token_grp:
        bucket = token_grp[key]
        if "DATA" not in bucket or "ID" not in bucket:
            continue
        sel = _select_rows(bucket, element_ids)
        if sel is None:
            continue
        rows, sel_ids = sel
        ncol = int(bucket["DATA"].shape[2])
        if ncol_ref is None:
            ncol_ref = ncol
            ncol_key = key
        elif ncol != ncol_ref:
            dropped.append((key, ncol, sel_ids))
            continue
        # (T, E_sel, ncol): only the requested steps and rows are read.
        block = read_hyperslab(bucket["DATA"], t_idx, rows)
        values_parts.append(block)
        eid_parts.append(sel_ids)

    if dropped:
        lost = np.concatenate([ids for _, _, ids in dropped])
        shown = ", ".join(str(int(i)) for i in lost[:8])
        if lost.size > 8:
            shown += f", ... (+{lost.size - 8} more)"
        widths = ", ".join(f"{k!r} has {n}" for k, n, _ in dropped)
        warnings.warn(
            f"Element read of token {token!r} kept the {ncol_ref}-column "
            f"bucket {ncol_key!r} and DROPPED {len(dropped)} bucket(s) of a "
            f"different width ({widths}); {lost.size} element(s) are missing "
            f"from the result: {shown}. Buckets of differing component count "
            f"cannot share one dense (T, E, ncol) slab, so the first width "
            f"wins. Read the differing elements in a separate call scoped to "
            f"them with element_ids= to get their columns.",
            ElementWidthMismatchWarning,
            stacklevel=2,
        )
    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(eid_parts),
    )


# =====================================================================
# Canonical nodal resisting forces (decoded from ``globalForce``)
# =====================================================================
#
# A ``nodal_resisting_{force,moment}_{x,y,z}`` request reads the
# ``globalForce`` token, the token MPCO reads for the same names, and
# names each column from the bucket's ``COMP_NAMES``. OpenSees spells the
# per-node columns of that response three ways:
#
# * axis-named ``Px_1, Py_1, Pz_1, Mx_1, My_1, Mz_1, Px_2, …`` (beams);
# * ``P<dof>_<node>``: ``P1_1, P2_1, P1_2, P2_2, …`` (quads, bricks, tets);
# * ``P<node>_<dof>``: ``P1_1, P1_2, P2_1, P2_2, …`` (trusses, zeroLength).
#
# The two indexed spellings share one grammar, so the column order decides
# which index is the node. An element's resisting-force vector is
# node-major: the node index runs 1, 1, …, 2, 2, … and the DOF index
# restarts at 1 for each node. A name list that reads node-major both ways
# (one node, or one DOF per node) is refused as ambiguous.
#
# An indexed DOF up to the spatial dimension is a translation. A higher
# indexed DOF means a rotation on a truss in a frame model, a pore pressure
# on a u-p element and an auxiliary field on a PML element; the names
# carry no hint which, so those columns stay undecoded, and a moment
# request that meets them warns.
#
# The slab is ``(T, E, n_nodes)`` in element-local node order, the layout
# :class:`MPCOReader` returns for the same names, so the two formats agree.

NODAL_FORCE_TOKEN = "globalForce"
_NODAL_GLOBAL = frozenset(
    c for c in PER_ELEMENT_NODAL_FORCES if "_local_" not in c
)
_AXIS_NAME_RE = re.compile(r"^([PM])([xyz])_(\d+)$")
_INDEXED_NAME_RE = re.compile(r"^P(\d+)_(\d+)$")
_AXES = ("x", "y", "z")


class NodalForceNotRecorded(MissingElementResults):
    """A canonical nodal resisting force the file is unable to answer.

    Raised by :func:`read_nodal_force_slab` when no ``globalForce`` bucket
    decodes to the requested name: the file does not record ``globalForce``,
    its buckets carry no such component (``nodal_resisting_moment_z`` on a
    truss), or the name is a local-frame one. The message names the tokens
    the file records and the canonical names it does answer.

    A subclass of :class:`MissingElementResults`, so a partitioned read
    tolerates a part file whose rank owns none of the elements and raises
    only when every part is unable to answer.
    """


class NodalForceColumnsUndecodedWarning(UserWarning):
    """``globalForce`` buckets were left out of a canonical nodal-force read.

    Raised by :func:`read_nodal_force_slab` for each bucket whose
    ``COMP_NAMES`` could not be decoded (an unknown spelling, an ambiguous
    order) or whose undecoded columns may hold the requested moment. The
    elements of those buckets are absent from the result.
    """


def _reads_node_major(pairs: "list[tuple[int, int]]") -> bool:
    """True when ``(dof, node)`` pairs run node 1..n, DOF 1..ndf each."""
    ndf: "Optional[int]" = None
    i = 0
    node = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][1] == node:
            j += 1
        dofs = [d for d, _ in pairs[i:j]]
        if not dofs or dofs != list(range(1, len(dofs) + 1)):
            return False
        if ndf is None:
            ndf = len(dofs)
        elif len(dofs) != ndf:
            return False
        i = j
        node += 1
    return ndf is not None


def decode_nodal_force_columns(
    names: "tuple[str, ...] | list[str]", *, ndm: int,
) -> "tuple[dict[str, ndarray], tuple[str, ...]]":
    """Name each ``globalForce`` column as a canonical nodal resisting force.

    Returns ``(columns, undecoded)``: ``columns`` maps each canonical name
    to its column index at each element node (node 1 first), and
    ``undecoded`` lists the indexed names above the spatial dimension
    ``ndm``, which are left out. Raises :class:`ValueError` with the reason
    when the names follow no nodal-force spelling, read node-major both
    ways or neither way, or miss a component at some node.
    """
    names = list(names)
    if not names:
        raise ValueError("the bucket names no columns")
    entries: list[tuple[Optional[str], int]] = []   # (canonical, node)
    undecoded: list[str] = []
    axis = [m for m in map(_AXIS_NAME_RE.match, names) if m is not None]
    indexed = [m for m in map(_INDEXED_NAME_RE.match, names) if m is not None]
    if len(axis) == len(names):
        for m in axis:
            kind = "force" if m.group(1) == "P" else "moment"
            entries.append(
                (f"nodal_resisting_{kind}_{m.group(2)}", int(m.group(3))),
            )
    elif len(indexed) == len(names):
        ab = [(int(m.group(1)), int(m.group(2))) for m in indexed]
        readings = [
            pairs for pairs in (ab, [(b, a) for a, b in ab])
            if _reads_node_major(pairs)
        ]
        if len(readings) != 1:
            raise ValueError(
                f"the indexed columns {names[:6]} read as node-major "
                f"{'both' if readings else 'neither'} way(s), so which "
                f"index is the node is unknown"
            )
        for name, (dof, node) in zip(names, readings[0]):
            if dof <= min(ndm, 3):
                entries.append(
                    (f"nodal_resisting_force_{_AXES[dof - 1]}", node),
                )
            else:
                entries.append((None, node))
                undecoded.append(name)
    else:
        bad = [n for n in names
               if not (_AXIS_NAME_RE.match(n) or _INDEXED_NAME_RE.match(n))]
        raise ValueError(
            f"the column(s) {bad[:4]} follow neither nodal-force spelling "
            f"(axis-named 'Px_1' or indexed 'P1_1')" if bad else
            f"the columns {names[:6]} mix the axis-named and indexed "
            f"spellings"
        )
    n_nodes = max(node for _, node in entries)
    per: dict[str, list[int]] = {}
    for col, (canonical, node) in enumerate(entries):
        if canonical is None:
            continue
        slots = per.setdefault(canonical, [-1] * n_nodes)
        if node < 1 or slots[node - 1] != -1:
            raise ValueError(
                f"{canonical} appears twice at element node {node}",
            )
        slots[node - 1] = col
    for canonical, slots in per.items():
        if -1 in slots:
            raise ValueError(
                f"{canonical} is absent at element node "
                f"{slots.index(-1) + 1}",
            )
    columns = {c: np.asarray(s, dtype=np.int64) for c, s in per.items()}
    return columns, tuple(undecoded)


def _decode_nodal_bucket(
    bucket: "h5py.Group", *, ndm: int,
) -> "tuple[dict[str, ndarray], tuple[str, ...]]":
    """:func:`decode_nodal_force_columns` over one bucket's element-level
    blocks; :class:`ValueError` when the bucket is not one."""
    for child in ("COLUMN_MAP", "DATA", "ID"):
        if child not in bucket:
            raise ValueError(f"the bucket has no {child}")
    blocks = parse_blocks(bucket)
    if any(b.level != 0 or b.multiplicity != 1 for b in blocks):
        raise ValueError("the bucket holds sub-element (Gauss) blocks")
    names = [n for b in blocks for n in b.comp_names]
    return decode_nodal_force_columns(names, ndm=ndm)


def indexed_nodal_force_buckets(on_elements: "h5py.Group") -> list[str]:
    """The ``globalForce`` bucket keys spelled in indexed ``P<a>_<b>``
    columns: the only spelling whose decode depends on ``ndm``.

    An axis-named bucket (``Px_1``, ``Mz_1``) names its quantity in the
    column, and a mixed or unparseable bucket is refused at any ``ndm``,
    so neither is listed.
    """
    if NODAL_FORCE_TOKEN not in on_elements:
        return []
    token_grp = on_elements[NODAL_FORCE_TOKEN]
    out: list[str] = []
    for key in token_grp:
        try:
            blocks = parse_blocks(token_grp[key])
        except (KeyError, ValueError):
            continue
        names = [n for b in blocks for n in b.comp_names]
        if names and all(_INDEXED_NAME_RE.match(n) for n in names):
            out.append(str(key))
    return out


def _nodal_force_available(
    on_elements: "h5py.Group", *, ndm: int,
) -> set[str]:
    """Canonical nodal resisting forces some ``globalForce`` bucket decodes."""
    out: set[str] = set()
    if NODAL_FORCE_TOKEN not in on_elements:
        return out
    token_grp = on_elements[NODAL_FORCE_TOKEN]
    for key in token_grp:
        try:
            columns, _ = _decode_nodal_bucket(token_grp[key], ndm=ndm)
        except (KeyError, ValueError):
            continue
        out.update(columns)
    return out


def read_nodal_force_slab(
    on_elements: "h5py.Group",
    canonical: str,
    *,
    ndm: int,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray]]":
    """Return ``(values[T, E, n_nodes], element_ids)`` for ``canonical``.

    ``canonical`` is a ``nodal_resisting_{force,moment}_{x,y,z}`` name,
    decoded from the ``globalForce`` buckets by
    :func:`decode_nodal_force_columns`. ``None`` when the id filter keeps
    no row. Raises :class:`NodalForceNotRecorded` when no bucket decodes
    to ``canonical``; warns :class:`NodalForceColumnsUndecodedWarning` for
    each bucket left out, and :class:`ElementWidthMismatchWarning` when
    buckets of a different node count are dropped (the first count wins).
    """
    recorded = sorted(str(k) for k in on_elements)
    if canonical not in _NODAL_GLOBAL:
        raise NodalForceNotRecorded(
            f"Element component {canonical!r} is a local-frame nodal "
            f"resisting force; this reader decodes only the global-frame "
            f"names, from {NODAL_FORCE_TOKEN!r}. The file records the "
            f"element responses {recorded}: read one of them by its token."
        )
    if NODAL_FORCE_TOKEN not in on_elements:
        raise NodalForceNotRecorded(
            f"Element component {canonical!r} is decoded from the element "
            f"response {NODAL_FORCE_TOKEN!r}, which this file does not "
            f"record. It records {recorded}: record {NODAL_FORCE_TOKEN!r}, "
            f"or read one of the recorded responses by its token."
        )
    token_grp = on_elements[NODAL_FORCE_TOKEN]
    answered = False
    offered: set[str] = set()
    skipped: list[tuple[str, str]] = []
    values_parts: list[ndarray] = []
    eid_parts: list[ndarray] = []
    n_ref: "Optional[int]" = None
    key_ref: "Optional[str]" = None
    dropped: list[tuple[str, int, ndarray]] = []

    for key in token_grp:
        bucket = token_grp[key]
        try:
            columns, undecoded = _decode_nodal_bucket(bucket, ndm=ndm)
        except (KeyError, ValueError) as exc:
            skipped.append((key, str(exc)))
            continue
        offered.update(columns)
        if canonical not in columns:
            if undecoded and "_moment_" in canonical:
                skipped.append((key, (
                    f"its columns {list(undecoded)[:6]} sit above the "
                    f"{ndm}-D translations, and an indexed column there "
                    f"may be a rotation, a pore pressure or an auxiliary "
                    f"field"
                )))
            continue
        answered = True
        sel = _select_rows(bucket, element_ids)
        if sel is None:
            continue
        rows, sel_ids = sel
        cols = columns[canonical]
        if n_ref is None:
            n_ref, key_ref = cols.size, key
        elif cols.size != n_ref:
            dropped.append((key, int(cols.size), sel_ids))
            continue
        values_parts.append(read_hyperslab(bucket["DATA"], t_idx, rows, cols))
        eid_parts.append(sel_ids)

    if skipped:
        detail = "; ".join(f"{k!r}: {why}" for k, why in skipped)
        warnings.warn(
            f"Element component {canonical!r} left out {len(skipped)} "
            f"{NODAL_FORCE_TOKEN!r} bucket(s) whose columns could not be "
            f"named ({detail}). Their elements are absent from the result; "
            f"read {NODAL_FORCE_TOKEN!r} by its token for their raw columns.",
            NodalForceColumnsUndecodedWarning,
            stacklevel=3,
        )
    if not answered:
        raise NodalForceNotRecorded(
            f"Element component {canonical!r}: no {NODAL_FORCE_TOKEN!r} "
            f"bucket in this file decodes to it. Its buckets decode to "
            f"{sorted(offered)}; the file records the element responses "
            f"{recorded}."
        )
    if dropped:
        lost = np.concatenate([ids for _, _, ids in dropped])
        shown = ", ".join(str(int(i)) for i in lost[:8])
        if lost.size > 8:
            shown += f", ... (+{lost.size - 8} more)"
        counts = ", ".join(f"{k!r} has {n}" for k, n, _ in dropped)
        warnings.warn(
            f"Element component {canonical!r} kept the {n_ref}-node bucket "
            f"{key_ref!r} and DROPPED {len(dropped)} bucket(s) of a "
            f"different node count ({counts}); {lost.size} element(s) are "
            f"missing from the result: {shown}. Read them in a separate "
            f"call scoped to them with element_ids=.",
            ElementWidthMismatchWarning,
            stacklevel=3,
        )
    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(eid_parts),
    )


# =====================================================================
# Fiber reads (fiber-section stress / strain)
# =====================================================================
#
# A ``.ladruno`` serialises a fiber section's per-fiber state under
# ``ON_ELEMENTS/section.fiber.stress`` (``…strain``) with one ``LEVELS==4``
# block per integration station whose ``MULTIPLICITY`` is the fiber count
# and ``NUM_COMP==1`` (the scalar uniaxial ``sigma11`` / ``eps11``). Fiber
# *geometry* (y, z, area, material) lives in
# ``MODEL/SECTION_ASSIGNMENTS/SECTION_<tag>/{FIBER_DATA, FIBER_MATERIALS}``,
# wired to (element, gauss) by that group's ``ASSIGNMENT[(elem_tag,
# gauss_id)]``. A layered shell writes under the same tokens, but each of
# its blocks carries several components per layer (``NUM_COMP`` 5 for a
# ``PlateFiber`` layer): those buckets are layer state, read by
# :func:`read_layer_slab` (see "Layer reads" below), and the fiber level
# keeps only the scalar ``NUM_COMP == 1`` blocks.


def _scalar_fiber_blocks(bucket: "h5py.Group") -> list[_Block]:
    """The bucket's per-fiber scalar blocks (``NUM_COMP == 1``), or ``[]``
    for a bucket whose ``COLUMN_MAP`` does not parse."""
    try:
        blocks = parse_blocks(bucket)
    except (KeyError, ValueError):
        return []
    return [b for b in blocks if b.gauss_id >= 0 and len(b.comp_names) == 1]


def fiber_available(on_elements: "h5py.Group") -> set[str]:
    """``fiber_stress`` / ``fiber_strain`` that some bucket carries as
    scalar per-fiber blocks. A token whose buckets are all layered-shell
    state (several components per layer) is listed at the layer level
    instead (:func:`layer_available`), since the fiber read returns none
    of it (#1995)."""
    out: set[str] = set()
    for token in on_elements:
        canonical = _FIBER_TOKEN_TO_CANONICAL.get(token)
        if canonical is None or canonical in out:
            continue
        token_grp = on_elements[token]
        if any(_scalar_fiber_blocks(token_grp[key]) for key in token_grp):
            out.add(canonical)
    return out


def _build_fiber_assignment(
    section_assignments: "Optional[h5py.Group]",
) -> "dict[tuple[int, int], tuple[ndarray, ndarray, ndarray, ndarray]]":
    """``(elem_tag, gauss_id)`` → ``(y, z, area, material_tag)`` per fiber."""
    out: dict[tuple[int, int], tuple[ndarray, ndarray, ndarray, ndarray]] = {}
    if section_assignments is None:
        return out
    for name in section_assignments:
        sa = section_assignments[name]
        if "FIBER_DATA" not in sa or "ASSIGNMENT" not in sa:
            continue
        fd = np.asarray(sa["FIBER_DATA"][...], dtype=np.float64).reshape(-1, 3)
        y, z, area = fd[:, 0], fd[:, 1], fd[:, 2]
        if "FIBER_MATERIALS" in sa:
            mats = np.asarray(
                sa["FIBER_MATERIALS"][...], dtype=np.int64,
            ).flatten()
        else:
            mats = np.full(fd.shape[0], -1, dtype=np.int64)
        assign = np.asarray(sa["ASSIGNMENT"][...], dtype=np.int64).reshape(-1, 2)
        for etag, gid in assign:
            out[(int(etag), int(gid))] = (y, z, area, mats)
    return out


def read_fiber_slab(
    on_elements: "h5py.Group",
    model_elements: "Optional[h5py.Group]",
    section_assignments: "Optional[h5py.Group]",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
    gp_indices: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumF], elem_index, gp_index, station_xi, y, z, area, mat)``.

    One slab column per (element, gauss point, fiber). ``None`` if the
    component (``fiber_stress`` / ``fiber_strain``) is absent.
    ``station_xi`` is the per-row station natural coordinate from the
    element's ``QUADRATURE/GP_PARAM`` keyed by ``GAUSS_ID`` (same
    source as the line-stations path); NaN for buckets without a
    recorded quadrature (e.g. layered shells, where the gauss id is a
    surface GP, not a beam station).
    """
    tokens = _FIBER_CANONICAL_TO_TOKENS.get(component)
    if tokens is None:
        return None
    present = [t for t in tokens if t in on_elements]
    if not present:
        return None
    assignment = _build_fiber_assignment(section_assignments)
    want_gp = None if gp_indices is None else set(int(g) for g in gp_indices)

    values_parts: list[ndarray] = []
    ei_parts: list[ndarray] = []
    gp_parts: list[ndarray] = []
    xi_parts: list[ndarray] = []
    y_parts: list[ndarray] = []
    z_parts: list[ndarray] = []
    area_parts: list[ndarray] = []
    mat_parts: list[ndarray] = []

    # Gather from every present spelling — a model with both fiber-section
    # beams (``section.fiber.*``) and layered shells (``material.fiber.*``)
    # emits both buckets.
    for token in present:
        token_grp = on_elements[token]
        # Beam-station natural coords only make sense for the beam
        # spelling — a layered shell's gauss_id is a SURFACE GP, so
        # its GP_PARAM (if any) is not a station along a member.
        is_beam_token = token.startswith("section.fiber")
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            fiber_blocks = [
                b for b in blocks
                if b.gauss_id >= 0 and len(b.comp_names) == 1
            ]
            # Blocks of several components per fiber are layered-shell
            # state, read at the layer level; a bucket that level is
            # unable to read warns here, by name.
            _layer_layout_or_warn(token, key, blocks)
            if not fiber_blocks:
                continue
            sel = _select_rows(bucket, element_ids)
            if sel is None:
                continue
            rows, sel_ids = sel
            # Hyperslab: the requested steps and rows, and only the
            # columns of the fiber blocks this read keeps.
            kept = [
                b for b in fiber_blocks
                if want_gp is None or b.gauss_id in want_gp
            ]
            if not kept:
                continue
            fib_cols = np.concatenate([
                np.arange(b.col_start, b.col_start + b.multiplicity)
                for b in kept
            ])
            data = read_hyperslab(bucket["DATA"], t_idx, rows, fib_cols)
            col_base: dict[int, int] = {}
            pos = 0
            for b in kept:
                col_base[id(b)] = pos
                pos += b.multiplicity
            gp_param = (
                _gp_param_for(model_elements, key) if is_beam_token
                else None
            )

            for b in sorted(fiber_blocks, key=lambda bb: bb.gauss_id):
                if want_gp is not None and b.gauss_id not in want_gp:
                    continue
                nfib = b.multiplicity
                # (T, E, nfib) — NUM_COMP==1, so the block is fiber-major.
                p0 = col_base[id(b)]
                block = data[:, :, p0:p0 + nfib]
                T = block.shape[0]
                E = sel_ids.size
                values_parts.append(block.reshape(T, E * nfib))
                ei_parts.append(np.repeat(sel_ids, nfib))
                gp_parts.append(np.full(E * nfib, b.gauss_id, dtype=np.int64))
                if (
                    gp_param is not None
                    and 0 <= b.gauss_id < gp_param.shape[0]
                ):
                    xi_val = float(gp_param[b.gauss_id, 0])
                else:
                    xi_val = np.nan
                xi_parts.append(np.full(E * nfib, xi_val, dtype=np.float64))
                # Per-element fiber geometry from the assigned section.
                ys = np.empty(E * nfib, dtype=np.float64)
                zs = np.empty(E * nfib, dtype=np.float64)
                ars = np.empty(E * nfib, dtype=np.float64)
                mts = np.empty(E * nfib, dtype=np.int64)
                for e_i, etag in enumerate(sel_ids):
                    geom = assignment.get((int(etag), b.gauss_id))
                    sl = slice(e_i * nfib, (e_i + 1) * nfib)
                    if geom is not None and geom[0].size == nfib:
                        ys[sl], zs[sl], ars[sl], mts[sl] = geom
                    else:
                        ys[sl] = np.nan
                        zs[sl] = np.nan
                        ars[sl] = np.nan
                        mts[sl] = -1
                y_parts.append(ys)
                z_parts.append(zs)
                area_parts.append(ars)
                mat_parts.append(mts)

    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(ei_parts),
        np.concatenate(gp_parts),
        np.concatenate(xi_parts),
        np.concatenate(y_parts),
        np.concatenate(z_parts),
        np.concatenate(area_parts),
        np.concatenate(mat_parts),
    )


# =====================================================================
# Layer reads (layered-shell stress / strain)
# =====================================================================
#
# A layered shell (``LayeredShellFiberSection`` on a shell) writes its
# per-layer state under the fiber tokens (``section.fiber.<resp>``, or
# ``material.fiber.<resp>`` when the request names it so), one
# ``LEVELS == 4`` block per SURFACE Gauss point whose ``MULTIPLICITY`` is
# the layer count and whose ``NUM_COMP`` is the layer material's component
# count (5 for ``PlateFiber``: 11, 22, 12, 23, 13). Inside a block the
# columns run layer-major, component-fast. The layer thickness and
# material sit in ``MODEL/SECTION_ASSIGNMENTS`` as for a fiber section
# (``FIBER_DATA`` column 2 is the thickness), assigned per ``(element,
# surface GP)``. The fork-lane test ``test_fiber_layer_readback_live.py``
# holds this layout against ``eleResponse``.
#
# The slab is the MPCO reader's ``LayerSlab`` (``_mpco_layer_io``):
# ``fiber_stress_<c>`` / ``fiber_strain_<c>``, rows element-slow, surface
# GP, then layer, sub-GP 0, the layer thickness per row, and the shell
# frame per row. A ``.ladruno`` stores no shell ``LOCAL_AXES``, so the
# frame is computed from the file's connectivity and node coordinates with
# :func:`~apeGmsh.results._shell_geometry.shell_quaternion`, as the MPCO
# reader does; a row with no frame reads NaN and warns by name.


class LayerSectionUnresolved(ValueError):
    """A layered-shell row has no layer thickness: its ``(element, surface
    GP)`` pair has no ``MODEL/SECTION_ASSIGNMENTS`` entry, or the assigned
    section's layer count differs from the bucket's. Names the bucket and
    the pairs (#1995)."""


#: A layer component and its parent: ``fiber_stress_3`` → ``fiber_stress``, 3.
_LAYER_COMPONENT_RE = re.compile(r"^(fiber_stress|fiber_strain)_(\d+)$")


@dataclass(frozen=True)
class _LayerLayout:
    """A layered-shell bucket: one block per surface GP, sorted by GP."""
    blocks: tuple[_Block, ...]
    n_layers: int
    n_comp: int

    def column(self, sgp: int, layer: int, comp: int) -> int:
        """DATA column of component ``comp`` of ``layer`` at surface GP
        position ``sgp``: layer-major, component-fast inside the block."""
        return self.blocks[sgp].col_start + layer * self.n_comp + comp


def _layer_layout(
    key: str, blocks: "list[_Block]",
) -> "tuple[Optional[_LayerLayout], Optional[str]]":
    """Classify a fiber-token bucket's blocks as layered-shell state.

    ``(layout, None)`` for a layered-shell bucket; ``(None, None)`` for a
    bucket with no block of several components per fiber (a scalar fiber
    bucket, read at the fiber level); ``(None, reason)`` for one whose
    multi-component blocks this reader does not decode.
    """
    multi = [b for b in blocks if b.gauss_id >= 0 and len(b.comp_names) > 1]
    if not multi:
        return None, None
    if len(multi) != len(blocks):
        return None, (
            f"{len(blocks) - len(multi)} of its {len(blocks)} blocks are "
            f"element-level or scalar beside the multi-component ones"
        )
    class_name = _class_name(key)
    if not is_layer_catalogued(class_name, "fiber_stress"):
        return None, (
            f"class {class_name!r} has no layered-shell entry in the "
            f"response catalog (LAYER_CATALOG)"
        )
    widths = {len(b.comp_names) for b in multi}
    layers = {b.multiplicity for b in multi}
    if len(widths) != 1 or len(layers) != 1:
        return None, (
            f"its blocks disagree on the components per layer "
            f"{sorted(widths)} or the layer count {sorted(layers)}"
        )
    ordered = tuple(sorted(multi, key=lambda b: b.gauss_id))
    gauss = [b.gauss_id for b in ordered]
    if gauss != list(range(len(ordered))):
        return None, (
            f"its surface Gauss ids {gauss} are not 0..{len(gauss) - 1}"
        )
    return _LayerLayout(
        blocks=ordered, n_layers=layers.pop(), n_comp=widths.pop(),
    ), None


def _layer_layout_or_warn(
    token: str, key: str, blocks: "list[_Block]",
) -> "Optional[_LayerLayout]":
    """:func:`_layer_layout`, warning by name for a bucket it leaves unread."""
    layout, reason = _layer_layout(key, blocks)
    if reason is not None:
        warnings.warn(
            f".ladruno ON_ELEMENTS/{token}/{key}: {reason}, so its fiber "
            f"blocks of several components are read at neither the fiber "
            f"nor the layer level.",
            LayerBucketUnreadWarning, stacklevel=3,
        )
    return layout


def _layer_buckets(
    on_elements: "h5py.Group", parent: str,
) -> "list[tuple[str, str, h5py.Group, _LayerLayout]]":
    """``(token, key, bucket, layout)`` per layered-shell bucket of
    ``parent`` (``fiber_stress`` / ``fiber_strain``), over both spellings."""
    out: list[tuple[str, str, h5py.Group, _LayerLayout]] = []
    for token in _FIBER_CANONICAL_TO_TOKENS[parent]:
        if token not in on_elements:
            continue
        token_grp = on_elements[token]
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            layout = _layer_layout_or_warn(token, key, blocks)
            if layout is not None:
                out.append((token, key, bucket, layout))
    return out


def layer_available(on_elements: "h5py.Group") -> set[str]:
    """``fiber_stress_<c>`` / ``fiber_strain_<c>`` per layered-shell bucket,
    one per component of the layer material (the MPCO reader's names)."""
    out: set[str] = set()
    for parent in ("fiber_stress", "fiber_strain"):
        for _t, _k, _b, layout in _layer_buckets(on_elements, parent):
            out.update(f"{parent}_{c}" for c in range(layout.n_comp))
    return out


def _index_filter(
    indices: "Optional[ndarray]", n: int, what: str,
) -> ndarray:
    """``indices`` as int64, every one in ``[0, n)``; all of them for None."""
    if indices is None:
        return np.arange(n, dtype=np.int64)
    sel = np.asarray(indices, dtype=np.int64).ravel()
    if sel.size and (int(sel.max()) >= n or int(sel.min()) < 0):
        raise ValueError(f"{what} {sel.tolist()} out of range [0, {n}).")
    return sel


def _model_child(
    model: "Optional[h5py.Group]", name: str,
) -> "Optional[h5py.Group]":
    return model[name] if model is not None and name in model else None


def _shell_frames(
    model: "Optional[h5py.Group]", key: str, element_ids: ndarray,
) -> ndarray:
    """Per-element shell frame, scalar-first ``(w, x, y, z)``, ``(n, 4)``.

    A frame the file stores (``MODEL/LOCAL_AXES/<classTag>-<Class>``)
    wins; every other element's frame is computed from the file's
    connectivity and node coordinates with :func:`shell_quaternion`, the
    convention the MPCO reader and DomainCapture use. A row with neither
    is NaN, and one :class:`ShellFrameUnavailableWarning` names its
    elements and why. A stored group without the ``ID`` + ``FRAME`` pair,
    or whose ``FRAME`` size is not four per ``ID``, is ignored with a
    :class:`ShellFrameUnavailableWarning` naming it, as the MPCO reader's
    ``_stored_quaternions`` does, and the frames are computed (#2090).
    """
    import h5py

    n = element_ids.size
    out = np.full((n, 4), np.nan, dtype=np.float64)
    prefix = _class_prefix(key)
    class_name = _class_name(key)
    failed: dict[str, list[int]] = {}

    def _fail(reason: str, eids: "list[int]") -> None:
        failed.setdefault(reason, []).extend(eids)

    stored: dict[int, ndarray] = {}
    axes = _model_child(model, "LOCAL_AXES")
    if axes is not None and prefix in axes:
        sub = axes[prefix]
        if "ID" not in sub or "FRAME" not in sub:
            warnings.warn(
                f".ladruno {sub.name}: expected ID and FRAME datasets, found "
                f"{sorted(sub)}; the stored frames are ignored and {class_name} "
                f"shell frames are computed from node coordinates.",
                ShellFrameUnavailableWarning, stacklevel=3,
            )
        else:
            ids = np.asarray(sub["ID"][...], dtype=np.int64).ravel()
            frames = np.asarray(sub["FRAME"][...], dtype=np.float64)
            if frames.size == 4 * ids.size:
                stored = dict(zip(ids.tolist(), frames.reshape(-1, 4)))
            else:
                warnings.warn(
                    f".ladruno {sub.name}: {ids.size} IDs but FRAME has shape "
                    f"{frames.shape}; the stored frames are ignored and "
                    f"{class_name} shell frames are computed from node "
                    f"coordinates.",
                    ShellFrameUnavailableWarning, stacklevel=3,
                )
    todo: list[int] = []
    for k in range(n):
        q = stored.get(int(element_ids[k]))
        if q is None:
            todo.append(k)
        else:
            out[k, :] = q
    if not todo:
        return out

    pending = [int(element_ids[k]) for k in todo]
    coords = np.zeros((0, 3), dtype=np.float64)
    node_row: dict[int, int] = {}
    nodes = _model_child(model, "NODES")
    if nodes is None or "ID" not in nodes or "COORDINATES" not in nodes:
        _fail("no MODEL/NODES ID and COORDINATES", pending)
        todo = []
    else:
        coords = np.asarray(nodes["COORDINATES"][...], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            _fail(f"node coordinates of shape {coords.shape}, not (n, 3)",
                  pending)
            todo = []
        node_row = {
            int(t): i for i, t in enumerate(
                np.asarray(nodes["ID"][...], dtype=np.int64).ravel())
        }
    conn_rows: dict[int, ndarray] = {}
    elements = _model_child(model, "ELEMENTS") if todo else None
    for name, grp in elements.items() if elements is not None else ():
        if (_class_prefix(name) != prefix or not isinstance(grp, h5py.Group)
                or "CONNECTIVITY" not in grp):
            continue
        conn = np.asarray(grp["CONNECTIVITY"][...], dtype=np.int64)
        if conn.ndim == 2 and conn.shape[1] >= 2:
            conn_rows.update((int(r[0]), r[1:]) for r in conn)
    for k in todo:
        eid = int(element_ids[k])
        tags = conn_rows.get(eid)
        if tags is None:
            _fail(f"no {prefix} row in MODEL/ELEMENTS", [eid])
            continue
        if any(int(t) not in node_row for t in tags):
            _fail("a node missing from MODEL/NODES", [eid])
            continue
        try:
            out[k, :] = shell_quaternion(
                coords[[node_row[int(t)] for t in tags]], class_name,
            )
        except ValueError as exc:
            _fail(str(exc), [eid])
    if failed:
        detail = "; ".join(
            f"{reason}: elements {sorted(set(eids))[:8]}"
            + (" ..." if len(set(eids)) > 8 else "")
            for reason, eids in failed.items()
        )
        warnings.warn(
            f".ladruno {class_name} layered-shell frames read NaN: the file "
            f"stores no LOCAL_AXES row for them and the frame cannot be "
            f"computed from the file's connectivity and node coordinates "
            f"({detail}).",
            ShellFrameUnavailableWarning, stacklevel=3,
        )
    return out


def read_layer_slab(
    on_elements: "h5py.Group",
    model: "Optional[h5py.Group]",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
    gp_indices: "Optional[ndarray]",
    layer_indices: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumL], element_index, gp_index, layer_index,
    sub_gp_index, thickness, local_axes_quaternion[sumL, 4])``.

    The MPCO reader's layout (``_mpco_layer_io.read_layer_bucket_slab``):
    one column per (element, surface GP, layer, sub-GP 0), element-slow,
    sub-GP fastest. ``component`` is ``fiber_stress_<c>`` /
    ``fiber_strain_<c>``; ``None`` when no layered-shell bucket carries it.
    ``gp_indices`` / ``layer_indices`` outside the bucket's range raise
    ``ValueError``, as on MPCO. A row whose layer thickness the file does
    not give raises :class:`LayerSectionUnresolved`.
    """
    m = _LAYER_COMPONENT_RE.match(component)
    if m is None:
        return None
    parent, comp = m.group(1), int(m.group(2))
    buckets = [
        b for b in _layer_buckets(on_elements, parent) if comp < b[3].n_comp
    ]
    if not buckets:
        return None
    assignment = _build_fiber_assignment(
        _model_child(model, "SECTION_ASSIGNMENTS"),
    )

    parts: list[list[ndarray]] = [[] for _ in range(7)]
    for token, key, bucket, layout in buckets:
        sel = _select_rows(bucket, element_ids)
        if sel is None:
            continue
        rows, sel_ids = sel
        sgp = _index_filter(gp_indices, len(layout.blocks), "gp_indices")
        lyr = _index_filter(layer_indices, layout.n_layers, "layer_indices")
        if sgp.size == 0 or lyr.size == 0:
            continue
        cols = np.array(
            [layout.column(int(g), int(k), comp) for g in sgp for k in lyr],
            dtype=np.int64,
        )
        data = read_hyperslab(bucket["DATA"], t_idx, rows, cols)
        n_e, per_e = sel_ids.size, sgp.size * lyr.size
        gauss = np.array(
            [layout.blocks[int(g)].gauss_id for g in sgp], dtype=np.int64,
        )
        thickness = np.empty(n_e * per_e, dtype=np.float64)
        missing: list[tuple[int, int]] = []
        for i, etag in enumerate(sel_ids):
            for j, g in enumerate(gauss):
                geom = assignment.get((int(etag), int(g)))
                if geom is None or geom[2].size != layout.n_layers:
                    missing.append((int(etag), int(g)))
                    continue
                start = (i * sgp.size + j) * lyr.size
                thickness[start:start + lyr.size] = geom[2][lyr]
        if missing:
            raise LayerSectionUnresolved(
                f".ladruno ON_ELEMENTS/{token}/{key}: {len(missing)} "
                f"(element, surface GP) pairs have no "
                f"MODEL/SECTION_ASSIGNMENTS section of {layout.n_layers} "
                f"layers, so their layer thickness is unknown; first few: "
                f"{missing[:5]}."
            )
        quats = _shell_frames(model, key, sel_ids)
        parts[0].append(data.reshape(data.shape[0], n_e * per_e))
        parts[1].append(np.repeat(sel_ids, per_e).astype(np.int64))
        parts[2].append(np.tile(np.repeat(gauss, lyr.size), n_e))
        parts[3].append(np.tile(lyr, n_e * sgp.size))
        parts[4].append(np.zeros(n_e * per_e, dtype=np.int64))
        parts[5].append(thickness)
        parts[6].append(np.repeat(quats, per_e, axis=0))
    if not parts[0]:
        return None
    return (
        np.concatenate(parts[0], axis=1),
        np.concatenate(parts[1]),
        np.concatenate(parts[2]),
        np.concatenate(parts[3]),
        np.concatenate(parts[4]),
        np.concatenate(parts[5]),
        np.concatenate(parts[6], axis=0),
    )
