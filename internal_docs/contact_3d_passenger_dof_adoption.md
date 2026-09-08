# apeGmsh ← Ladruno 3-D interfaces on u-p nodes: adoption note (fork ADR 96)

Fork PR #808 (build `a240b9183`, 2026-09-07). Companion to
`contact_2d_adoption.md`; the source of record is the fork's
`96_ladruno_contact_passenger_dof_adr.md` and its adoption guide
`ladruno_apegmsh_adoption_guide_2026-09-07.md`.

## What the fork changed

The pressure DOF — or any DOF past the third — now rides as a *passenger*
on `zeroLength` and on every 3-D contact lane: never read, never written,
never coupled. This is not ADR 47 deferral 9; there is no gap flow and no
pressure penetration.

- **`zeroLength` in 3-D** accepts any pair with both ndf ≥ 3 outside the
  vanilla `(3,3)` / `(6,6)` table: `(3,4)`, `(4,3)`, `(4,4)`, `(3,6)`,
  `(6,4)`. The 6-slot translational core is scattered into an
  `ndf1 + ndf2` element. Only `-dir 1 2 3` exist on such a pair; a
  rotational `-dir` is refused ("passenger mode, ADR-96 … element
  disabled") and the element stays in the domain contributing nothing.
- **Vanilla's differing-dof case no longer crashes.** `(2,3)` and friends
  keep the "have differing dof at ends for ZeroLength" wording but the
  element is now inert instead of dereferencing NULL in `update()`. Any
  apeGmsh guard that modelled that as a crash now models a warning plus a
  dead element — which is why `_validate_interface_ndf` and
  `validate_adaptive_element_endpoints` in `opensees/_internal/build.py`
  keep refusing the mismatch *before* emission; the fork will not.
- **Reader contract.** The `force` response of such an element is
  **element-sized** (`ndf1 + ndf2` slots; node 2's translations at
  `ndf1 .. ndf1+2`, every passenger slot exactly `0.0`). `deformation`,
  `material` and `basicForce` stay the 6-slot core. Neither the MPCO spring
  reader (`basicForce`, width from `NUM_COMPONENTS`) nor the `.ladruno`
  element reader (width from `NUM_COLUMNS`) assumes six, so nothing on our
  side changes; a reader that ever hard-codes a 6-vector for `force` is
  wrong from this build on.
- **Contact, 3-D lane** (NTS, mortar, edge-edge, rigid plane): any node on
  either side may be ndf ≥ 3; the adapter takes each node's first three
  equations. Nothing changes in the emitted `contactSurface` / `contact`
  lines — u-p nodes may already be listed as slaves or masters.

## The 2-D lane is unchanged

`ndf == ndm == 2` exactly, on both contact and `zeroLength`. Everything in
`contact_2d_adoption.md` still holds, refusals included.

## What this does and does not unblock here

It clears blocker 1 of `plan_interface_3d.md` (TIMs A10). Blocker 2 —
per-facet frames, the surface tributary model and the second tangent of
ADR 0093 D2/D3 in 3-D — is ours and untouched, so `g.constraints.interface()`
still refuses a 3-D model at the call. The S1–S4 slices in that plan are
the route; S2 is where the phantom bridge learns the pairs above.

`LadrunoKinematicCoupling` is the other 3-D u-p touchpoint: fork #814 now
refuses an ambiguous slave ndf without `-dof` at the parser, which is the
same case the bridge's A1 gate (`build.py`, `_check_default_coupling_slave`)
refuses before emission. Both sides agree; the emitted command is unchanged.
