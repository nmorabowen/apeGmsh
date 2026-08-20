# apeGmsh ← Ladruno 2-D contact: adoption brief (fork ADR-85)

> ## STATUS — most of this brief is now HISTORY (2026-08-19)
>
> The adoption shipped: apeGmsh **#1048** and fork **#764**, plus **S4**
> (below). Read this header before believing anything below it — the body
> was written when apeGmsh could reach none of the lane, and §2's refusal
> table in particular now describes code that no longer refuses.
>
> **Working today.** 2-D NTS contact with friction, end to end: a chained
> stride-2 master surface built from a meshed dim-1 PG, 2-component
> `-outward`, `outward="winding"` (fork `-outward winding`, NTS only),
> and the rigid analytical plane. Acceptance: an apeGmsh 2-D two-body
> deck solves on the fork with equilibrium closing at machine precision,
> and winding returns bit-identical results to an explicit vector.
>
> **S4 shipped too — 2-D mortar, tie and `-thickness`.** `-slave-segments
> 2` reuses the master's chain walk and the same stride-2 record
> validator, so a holed slave listing (silently legal fork-side, exactly
> like a holed master) is unreachable by construction on both sides;
> `thickness=h` emits the mortar-only `-thickness`, refused by name on the
> NTS lane and in a 3-D model. Acceptance, measured on fork build
> `e7555f2c9` (T4): a flush 2-D mortar **tie** deck closes equilibrium at
> machine precision (residual 0 to 2.3e-10 on a 1e6 load), and a
> compression deck pins all three thickness conventions at once —
> no `-thickness` is bit-identical to `-thickness 1.0`; halving `h`
> against an EXPLICIT `-epsN` doubles the interface penetration part
> exactly (4.013e-06 both times, so `h` is applied ONCE, not squared); and
> `-epsN auto` under `h = 1.0` vs `h = 0.5` is **bit-identical**
> (5.244762212098e-05) — auto is never h-scaled.
>
> **S5 shipped — the docs.** `guide_constraints.md` gained **Level 5 —
> Contact** (the whole verb, 3-D and 2-D: it was documented nowhere
> before, not even in 3-D), the skill cheatsheet gained a 2-D contact
> section and a corrected source-map footer, and
> `docs/concepts/constraints.md` / `backend-capabilities.md` were
> reconciled. §4's curved-master facet-sizing warning now lives in the
> guide's *Meshing the interface* list rather than only on this
> history-flagged page, with a pointer from `guide_meshing.md` §5.
>
> **Not adopted yet.** **Results** — `Results/` reads no contact data at
> all (S6). Parallel/DDM 2-D contact is out of scope fork-side and is
> refused by name on both lanes.
>
> **The F1 asymmetry, carried not papered over.** `outward="winding"` is
> **NTS-only**. The fork shipped declared winding on the NTS lane alone
> because its 2-D mortar lane has no chain-integrity scan to rest
> winding's one-connected-chain invariant on, so a flush **mortar**
> interface — the workhorse 2-D case — still requires an explicit
> `outward=(ox, oy)`. `ContactDef` refuses the combination at
> declaration, and the flush refusal offers a mortar caller the vector
> alone, saying why. Two consequences follow that are easy to forget: a
> curved or closed **mortar** master stays undeclarable (only winding
> orients those), and the wrong-side master guard is apeGmsh's on BOTH
> lanes — the fork's centroid vote only picks a SIGN, so a far-side master
> resolves happily against a boundary the slave never reaches.
>
> **Three facts to carry, not re-derive.**
> 1. *Winding sign.* Every emitted segment satisfies
>    `dot(perp(t), normals[e]) == +1` — the slave lies to the LEFT of
>    chain travel. This shipped **inverted** first and was caught by
>    reading it, not by tests: 48 new tests passed over it because they
>    all checked structure and none checked which side the slave was on.
>    Pinned by `test_winding_puts_the_slave_on_the_left`.
> 2. *Coarse closed loops transmit exactly zero.* Fork-measured on a unit
>    n-gon: n ≤ 8 inert, n ≥ 9 exact. Disclosed and pinned fork-side, not
>    fixed. Say "closed loops work **if adequately refined**".
> 3. *A zero initial gap does not arm the NTS lane.* A body whose only
>    restraint is the contact then has a rigid-body mode and diverges
>    (measured increment norm 1.0e+12); seed a small overlap. The
>    **rigid-plane lane does not need this** — it arms from a zero gap.
>
> **Hazard left standing.** Two mechanisms now trim node coordinates:
> main's emitter-side `trim_coords_to_ndm` (ADR 0099, authoritative, runs
> last) and this work's build-layer `node_coords_for_ndm`. One invariant,
> two enforcement points — reconcile deliberately, not as merge fallout.
>
> §5's slice plan and `contact_2d_fork_asks.md` §B are both superseded by
> what actually shipped; S5 retires them.

The Ladruno fork closed its 2-D contact lane on 2026-08-18 (`e7555f2c9`,
"close the 2D contact lane — T0–T4 shipped"). Every 2-D lane is live: rigid
plane, NTS penalty with friction, mortar/ALM with friction and tie, and the
D4 radial end-cap. **apeGmsh cannot reach any of it.** Every layer of the
contact path — the facet extractor, the emit grammar, the H5 schema — is
hard-wired to 3-node/4-node facets, and one blocker sits *upstream* of
contact entirely, in the node emitter.

This page is the adoption map: what the fork gives us, exactly where
apeGmsh refuses it, and the order to close it in. It is not a user guide —
nothing here works yet. When the lane lands, the user-facing material
belongs in `guide_constraints.md` (which today has zero 2-D contact
content) and the fork's own `LadrunoContact2D_guide.md` stays the
authority on kernel behaviour.

Sources: fork ADR `85_ladruno_contact_2d_adr.md` (design record) and
`LadrunoContact2D_guide.md` (user guide), both in
`OpenSees/Ladruno_implementation/`. Read §8 "API and dimension routing" of
the ADR before touching the emit grammar.

## 1. What the fork shipped

The command surface, 2-D forms:

```tcl
contactSurface <tag> -master 2 <n0> <n1>  <n1> <n2>  ...   ;# stride-2 PAIR list
contactSurface <tag> -slave  <n0> <n1> ...                 ;# node set (NTS)
contactSurface <tag> -slave-segments 2 <n0> <n1>  <n1> <n2> ...

contact <tag> <master> <slave> <kn>|auto <kt> <mu> [-outward <ox> <oy>] ...
contact <tag> <master> <slave> "-mortar" -epsN <v>|auto [-thickness <h>] [-tie] ...

contactPlane <tag> <slaveSurf> <nx> <ny> <px> <py> <kn>          ;# 7-arg 2D form
contactPlane <tag> <slaveSurf> <nx> <ny> 0 <px> <py> 0 <kn>      ;# 9-arg, still valid
```

Three properties of the fork design drive everything below.

**The dimension oracle is node coordinates, not `ndm`.** The fork
deliberately refused to branch on interpreter `ndm` state (mutable, no
null guard on the Tcl path). `contactSurface` and `contactPlane` both
derive dimension from the referenced surface's node coordinates. So what
apeGmsh writes on the `node` line decides which lane the fork picks — see
§3.

**The guards are declaration-time and loud.** Every node of both surfaces
must carry the same `getCrds().Size() ∈ {2,3}`; that size must match the
declared arity (2 ⇔ `nps 2`, 3 ⇔ `nps 3/4`); and on the NTS/mortar/tie
lanes **`ndf == ndm` exactly**. Only the rigid-plane lane keeps
`ndf >= ndm`. Any violation is a named abort, never a silent DOF shift.

**`-outward` takes two components in 2-D** — and the 3-component 3-D form
is *rejected* on a 2-D surface. This inverts an apeGmsh policy; see §4.

## 2. Where apeGmsh refuses it

| Layer | Site | The 3-D assumption |
|---|---|---|
| Facet extraction | `core/ConstraintsComposite.py:129` | `_SURFACE_CORNER_NPS = {3:3, 6:3, 4:4, 8:4, 9:4}` — no key `2`. A line-element surface raises "not a supported tri or quad facet". |
| Emit grammar | `opensees/element/contact.py:61` | `_check_nps`: `nps not in (3, 4)` → `-master 2` is unreachable. |
| Outward | `opensees/element/contact.py:232` | `len(outward) != 3` raises — rejects the 2-component form the 2-D lane *requires*. |
| Rigid plane | `opensees/element/contact.py:264` | `len(n) != 3 or len(p) != 3` raises; always emits the 9-arg form. |
| Mortar thickness | `opensees/element/contact.py` (absent) | No `-thickness` parameter exists anywhere in the emitter. |
| H5 round-trip | `mesh/_femdata_h5_io.py:1904`, `:3656` | `master_nps` validated "expected 3 or 4" on both write and read — an `nps=2` record cannot survive `save` / `from_h5`. |
| H5 outward slot | `mesh/_record_h5.py:477`, `_femdata_h5_io.py:1930`, `:3679` | `("outward", np.float64, (3,))` — a fixed 3-slot payload, truncated `[:3]` on both write and read. A 2-component outward cannot round-trip. |
| Guide | `internal_docs/guide_constraints.md` | Zero 2-D contact content. |

The ownership resolver (`_kernel/resolvers/_contact_ownership.py`)
reshapes by `record.master_nps` generically and looks dimension-agnostic
— but it is partitioning-only, and **parallel/DDM 2-D contact is
explicitly out of scope in the fork ADR**. Do not spend effort there;
refuse 2-D under partitioning by name instead.

One lane already works, and not by accident: `contact_plane` emits the
9-arg zero-padded form, the rigid-plane lane skips the `ndm 3` pre-flight
(`LadrunoContactHandler.cpp:100` is called only from the NTS/mortar/tie
blocks), and fork T0 **gated** the behaviour — "9-arg `contactPlane` on a
2D surface still works (back-compat)" is a falsifier in G-T0(c), PR #749.
So the rigid-plane lane is tested back-compat, not a hopeful accident, and
it is the cheapest real 2-D contact in the library.

## 3. P0 — the blocker upstream of contact

**apeGmsh always emits three node coordinates**, unconditionally:

```python
# opensees/_internal/build.py:952
emitter.node(int(tag), float(x), float(y), float(z))
```

Measured against the fork (build `25a0647f`, 2026-08-18): in a 2-D deck
OpenSees consumes `ndm` coordinates, then the extra `0.0` desynchronises
the optional-argument scan and **silently swallows `-ndf`**:

```
model BasicBuilder -ndm 2 -ndf 3
node 1 0.0 0.0 0.0 -ndf 2   ->  node ndf 3   (override LOST)
node 2 1.0 0.0     -ndf 2   ->  node ndf 2
```

**This is the Tcl route only.** Measured on the same build: the
openseespy `ops.node` path parses the identical over-long argument list
correctly — `ops.node(1, 0.0, 0.0, 0.0, "-ndf", 2)` resolves to ndf 2. So
`PyEmitter` and live decks were never affected, and any regression gate
for this must run the real Tcl binary in a subprocess; a `LiveOpsEmitter`
test would pass on a broken build and prove nothing.

The node still lands with `getCrds().Size() == 2`, so the fork's dimension
oracle reads correctly — the coordinate itself is not the problem. The
problem is the **`ndf == ndm` guard**. Per-node ndf was inert in every 2-D
apeGmsh *Tcl* deck, so a mixed-ndf 2-D model (plane elements at ndf 2
sharing a model with anything at ndf 3) could never satisfy the
NTS/mortar/tie guard on that route:
apeGmsh emits `-ndf 2`, the token is present and inert, the node carries
ndf 3, and the fork aborts the contact declaration by name. The abort is
loud, but its cause is invisible from the deck — the token you would read
to diagnose it is exactly the one that did nothing.

**Fix this before any contact work.** Emit only `ndm` coordinates in a 2-D
deck. It is a small change at one site, it is a prerequisite for the
`ndf == ndm` guard to be satisfiable at all, and it independently repairs
per-node ndf for every 2-D model — contact or not.

## 4. Three hazards apeGmsh should own

These are places where apeGmsh generating the deck can remove a class of
error the hand-written Tcl user has to avoid by discipline. This is the
real argument for adopting the lane in apeGmsh rather than telling people
to hand-roll it.

**The chained stride-2 pair list — the highest-value item.** `-master 2`
takes a flat stride-2 pair list chained head-to-tail: three segments need
*six* tags, `101 102  102 103  103 104`. The natural-looking shorthand
`101 102 103 104` is **silently legal** and declares two disjoint segments
with a hole where the middle one should be. The fork cannot refuse it — an
even tag count with no repeat is indistinguishable from a genuinely
disjoint surface, which is also legitimate. The measured result is the
ADR-78 "converged, balanced, wrong" shape: master row forces
`[5/18, 8/18, 5/18]·P` instead of the intended `[1/4, 1/2, 1/4]·P` — the
exact discrete solution of the *holed* surface, not a numerical error.
apeGmsh builds this connectivity from a meshed PG, so it can make the hole
**impossible by construction**. Emit chained pairs from ordered edge
connectivity and the user never meets the trap. This is the single
strongest reason to route 2-D contact through apeGmsh.

**Flush interfaces invert our `-outward` policy.** apeGmsh deliberately
never auto-derives an outward normal — `core/ConstraintsComposite.py:522`,
"never auto-derive a global outward here", because the 3-D kernel computes
a correct per-facet normal and a single global outward would silently
invert facets opposed to it. That reasoning is sound *for 3-D* and does
not carry over. The 2-D lane derives orientation from an interface-level
centroid vote computed once at `handle()`, and when master and slave
centroids are coincident the vote is genuinely ambiguous and the deck
**aborts** unless `-outward ox oy` is supplied. Flush is not an edge case
in 2-D — it is the masonry joint, the footing seated on soil, every
zero-gap interface. So for 2-D, apeGmsh must detect the flush case and
either require `outward` from the caller with a named error or derive it
from geometry. Whichever way that goes, the 3-D "never auto-derive"
comment needs an explicit 2-D carve-out, or it will be read as settling a
question it never considered.

The require-versus-derive framing above is superseded now that fork-side
changes are on the table: a direction vector structurally cannot express
the correct side of a *curved* or closed master, so a declared-winding
channel beats both options. See ask 1 of `contact_2d_fork_asks.md`.

**Curved masters need facet length sized from penetration, not the mesh.**
On a faceted arc the NTS narrow phase stops arming a pair once penetration
exceeds roughly twice the local facet length, silently disarming the
interior of the contact patch and collapsing load onto a rim of surviving
nodes. It looks like ordinary discretization error. Worse, if the arc's
facet length is driven by the same mesh parameter as the surrounding
elastic mesh — the natural choice, and the one apeGmsh's sizing API makes
easiest — refining the mesh makes it *worse*. apeGmsh owns meshing, so it
is the right place to warn. The stronger practical limit on curved masters
is upstream of this, though: the orientation vote itself splits on a
strongly curved or inconsistently wound master and draws a named refusal
even when `-outward` *is* supplied, which makes a closed-loop master
undeclarable today. That is what ask 1 of `contact_2d_fork_asks.md`
targets.

## 5. Adoption order

Each slice ends with the fork's own gate idiom: the existing 3-D contact
battery green with an unchanged pass count, recorded in the PR body.
Nothing in the 3-D lane may move.

**S0 — the node-coordinate prerequisite (§3).** Emit `ndm` coordinates in
2-D decks. Gate: a mixed-ndf 2-D deck reports the ndf it was told to carry
(verified with `llength [nodeDisp $tag]`, not by reading the emitted
`-ndf` token — the token can be present and inert); 3-D decks
byte-identical.

**S1 — rigid plane.** No probe needed (§2): the 9-arg path is gated
back-compat fork-side. This slice is a test plus a `contact_plane_args`
that accepts 2-vectors. The cheapest real 2-D contact in the library, and
it keeps `ndf >= ndm`, so it does not depend on S0 landing first.

**S2 — the grammar and the pair list.** `nps=2` through `_check_nps`,
`_SURFACE_CORNER_NPS`, and the H5 schema; 2-component `-outward`; chained
pair-list generation from ordered edge connectivity (§4). Gate: a
three-segment master emits six tags in chained order, and a test asserts
the *holed* four-tag form is never produced.

**S3 — NTS with friction, and the flush rule.** The `outward` policy
carve-out, with a named error on an ambiguous flush interface.

**S4 — mortar, tie, `-thickness`.** SHIPPED; see the status header for
what was measured. Note the two independent thickness
conventions: element thickness is baked into element stiffness and contact
never re-reads it; mortar `-thickness h` scales `epsN`/`epsT`/`-visc`/the
friction clamps/tie stiffness once at injection; and **`-epsN auto` is not
h-scaled** (it already absorbs element thickness via `getInitialStiff()` —
re-scaling is an h² error, regression-gated fork-side). NTS has no
`-thickness` at all.

**S5 — docs.** SHIPPED; see the status header. `guide_constraints.md`
gained Level 5, and the curved-master warning (§4) landed in its *Meshing
the interface* list — the judgement call being that the reader who needs
that warning is the one declaring a curved master, not the one tuning
mesh sizes, so it sits with the contact verb and `guide_meshing.md` §5
only signposts it.

Refuse 2-D under partitioning by name throughout — out of scope fork-side.

## 6. Open before S2

Two of the three questions this section originally posed are now answered
— `contact_plane` is gated back-compat in 2-D (§2), and the flush
`-outward` choice is superseded by the declared-winding ask (§4). What
remains open:

- **Where do 2-D master segments come from?** A meshed 1-D PG (line
  elements on the interface), or edges derived from the 2-D solid mesh.
  The second avoids asking users to mesh an interface curve, but the
  ordering the chained pair list needs has to come from somewhere — the
  fork's `handle()` scan FATALs on a permuted or reversed listing.

## See also

- `contact_2d_fork_asks.md` — the fork-side change request, written once
  asking the fork team for changes became an option. It supersedes §5's
  slice plan with one that tags each slice apeGmsh-side / fork-side, and
  its §C records the corrections already folded back into this page.
- `Ladruno_implementation/LadrunoContact2D_guide.md` — kernel behaviour,
  vertex policy, the D4 end-cap, the thickness table. The authority.
- `Ladruno_implementation/85_ladruno_contact_2d_adr.md` §8 — dimension
  routing and the guard contract.
- `guide_constraints.md` — where the user-facing 2-D section lands.
- ADR 0073 (contact generator) and ADR 0092 (partitioned contact emit) —
  the apeGmsh-side decisions this brief extends.
