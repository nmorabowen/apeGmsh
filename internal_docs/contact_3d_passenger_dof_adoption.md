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

It cleared blocker 1 of `plan_interface_3d.md` (TIMs A10). Blocker 2 —
per-facet frames, the surface tributary model and the second tangent of
ADR 0093 D2/D3 in 3-D — was ours, and S1–S3 closed it: `interface()` on a
3-D surface master now resolves *and* emits, one `zeroLength` per pair
with `-mat mN mT mT -dir 1 2 3` and the pair's own `-orient`. The pairs
above are taken directly, with no phantom bridge — that table
(`_ACCEPTED_3D_NDF_PAIRS`) is declared once in the resolver and imported
by the emit-time gate. **S4 has landed too, so A10 is COMPLETE**: the
2-D acceptance case reproduces in 3-D to 1.4e-16 (settlement) / 1.6e-16
(normal-spring sum) on a one-element-deep twin, the three springs read
back per pair through `Results.from_mpco`, a master wrapping a convex
corner runs while its reentrant mirror is refused, and the two uncoupled
tangential sliders were measured at 1.40 on the diagonal against the
√2 the square locus predicts.

This note's own G2 assertion — the passenger DOF is neither read nor
written — is now reproduced at model scale, not only on the fork's hand
deck: on a `(4, 3)` pair under a real `LadrunoUP` soil with `p = 1e6`
imposed on an interface node, `eleResponse force` is 7 wide with the
master's DOF-4 slot exactly `0.0` on every pair, and the pore-pressure
field is identical (rel 1.5e-17) to the same model tied with
`equalDOF 1 2 3` instead — this note's G3 twin. A live smoke on a
`(4, 3)` u-p deck reads this note's three warnings out of the fork's log
and fails on them.

`LadrunoKinematicCoupling` is the other 3-D u-p touchpoint: fork #814 now
refuses an ambiguous slave ndf without `-dof` at the parser, which is the
same case the bridge's A1 gate (`build.py`, `_check_default_coupling_slave`)
refuses before emission. Both sides agree; the emitted command is unchanged.
