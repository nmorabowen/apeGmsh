# apeGmsh ← Ladruno 2-D contact: the fork-side change request

> **STATUS (2026-08-19).** Ask 1 (`-outward winding`) SHIPPED as fork
> PR #764 — NTS lane only, with the connected-chain guard §A argued
> for, plus a wire-format bump (`LCD_FMT_VERSION` 2→3) and one
> disclosed limitation: a coarse closed loop (n ≤ 8) transmits
> exactly zero. Asks 2 and 3 were NOT filed. **§B's plan is
> superseded** by what shipped — see the status header of
> `contact_2d_adoption.md`.

The adoption brief (`contact_2d_adoption.md`) was written under a constraint
that no longer holds: it assumed every gap between the fork's 2-D contact
lane and apeGmsh had to be closed on the apeGmsh side. We can now ask the
fork team for changes. This page is the re-derivation: the change request
itself (§A), the adoption plan revised for the new degree of freedom (§B,
superseding §5 of the brief), and a correctness pass on the brief (§C).

The request is deliberately short. Three asks, ranked. Two more candidates
were evaluated and dropped, with reasons — a rejected ask still costs the
fork team a review, so the bar is "apeGmsh cannot get this any other way,"
not "this would be nice."

Sources are the fork ADR (`85_ladruno_contact_2d_adr.md`), the user guide
(`LadrunoContact2D_guide.md`), and the shipped C++ read directly; every
cost estimate below was made against the actual parser and handler, not
the ADR's description of them.

## A. The change request

### Ask 1 — orientation from declared winding (`-outward winding`)

**The problem.** The 2-D NTS and mortar lanes take their normal sign from
an interface-level centroid vote: one σ per contact, chosen so the
per-segment normal `σ·perp(t)` aligns with `slave_centroid −
master_centroid`, unanimity required across master segments
(`LadrunoContactHandler.cpp:441-523`). Two refusals follow from that
design. A **flush** interface (coincident centroids — the masonry joint,
the footing on soil, every zero-gap deck this lane was built for) fails
the magnitude gate and aborts unless `-outward ox oy` is given
(`:494-509`). A **strongly curved** master splits the vote and aborts even
*with* `-outward`, because no single direction vector agrees with every
segment's perpendicular (`:510-519`); the guide's remedy is to split the
surface into separately declared contacts, and a closed-loop master (a
ring, a full indenter profile) cannot be oriented at all — every vector is
tangent to or opposed by some segment. On the apeGmsh side, our standing
policy is to never auto-derive a global outward vector
(`core/ConstraintsComposite.py:522`), and the 2-D vote reproduces exactly
the failure that policy exists for: one global direction silently
mis-serving a curved surface.

**The proposed change.** Accept a declared-winding orientation mode —
suggested spelling `-outward winding`, mirroring the `kn auto` keyword
idiom — meaning: σ = +1, i.e. every master segment's normal is
`perp(t) = (−t_y, t_x)` of its own travel direction
(`LadrunoContact2DKernel.h:156-157`), the slave side lying to the left of
the chain's traversal. The centroid vote is bypassed entirely. The
invariant this relies on — the master segments form one consistently
wound head-to-tail chain — is *already enforced*: the handle()-time
chain-integrity scan FATALs by name on any permuted, reversed, or forked
listing (`LadrunoContactHandler.cpp:1193-1240`). Winding is per-segment
exact at any curvature, so the split-vote refusal and the closed-loop
impossibility both disappear for decks that declare it.

**Why fork-side beats an apeGmsh workaround.** apeGmsh knows the correct
side without guessing: a boundary edge of a consistently oriented 2-D
solid element has material on a known side of its traversal direction, so
the generator can wind master chains from mesh adjacency deterministically.
But the only channel that knowledge fits through today is a direction
*vector*, which structurally cannot represent a curved or closed master.
The information apeGmsh has is a winding; the fork accepts only a vector.
No apeGmsh-side effort converts one into the other in the cases that
matter.

**Cost, read from the code.** Small. One branch in the `contact` parser's
option loop (the existing `-outward` arity branch at
`OpenSeesOutputCommands.cpp:861-925` is the template — peek the token,
`"winding"` is not a number), a flag through the contact struct, and a
short-circuit at the two `ladruno2DOrientationVote` call sites
(`LadrunoContactHandler.cpp:1248`, `:2030`) setting σ = +1 instead of
voting. The vote helper, the kernel, the vertex/end-cap machinery (which
consume σ uniformly), and every 3-D path are untouched; `contact_dump`
byte-identity is preserved trivially. The one real design obligation is
documenting the sign convention once, precisely. The honest counter the
fork team may raise: winding intent is less self-documenting in a deck
than a vector. But chain order is *already* load-bearing (the concave
vertex pairs and the ownership stand-down key off it) and already
enforced — the flag makes an existing reliance explicit rather than
adding a new one.

**If the answer is no.** apeGmsh requires `outward=` from the caller on
every flush 2-D interface, with a named error that names both surfaces
and shows the call to add; strongly curved masters are refused with
split-the-surface guidance; closed-loop masters remain undeclarable. The
lane still ships — the workhorse flat-interface decks only suffer one
extra required argument — but a real capability gap stays open and is
documented as fork-inherited.

### Ask 2 — a chained declaration form (`-master-chain` / `-slave-chain`)

**The problem.** `-master 2` takes a flat stride-2 pair list chained
head-to-tail: three segments need six tags. The shorthand
`101 102 103 104` is silently legal and declares a *holed* surface —
measured `[5/18, 8/18, 5/18]·P` against the intended
`[1/4, 1/2, 1/4]·P`, the exact discrete solution of the holed surface
("converged, balanced, wrong"). The fork ADR argues the parser cannot
refuse this, and for the pair-list grammar that argument is airtight: an
even tag count with no repeats is indistinguishable from a genuinely
disjoint surface, which is also legitimate, and the chain-integrity scan
deliberately cannot refuse it either (no node is shared, so no rule
fires — `LadrunoContactHandler.cpp:1205-1240`).

**The proposed change.** A distinct declaration form sidesteps the
argument instead of fighting it: `contactSurface tag -master-chain n0 n1
… nk` (and the `-slave-chain` sibling for mortar) declares k segments
`(n0,n1), (n1,n2), …` — inside this form a hole is *unrepresentable*,
not merely discouraged. Genuinely disjoint surfaces keep the pair-list
form. The parser expands the chain into the existing stride-2 pair list
and constructs the same `LadrunoContactSurface` (`nps = 2`); the
chain-integrity scan is satisfied by construction.

**Why fork-side beats an apeGmsh workaround.** Partially, it doesn't —
and the ask says so. apeGmsh generates connectivity from a meshed PG and
will gate S2 on never emitting the holed form, which fully protects
apeGmsh decks. The residual value is real but second-order: an emit bug
producing a legal-but-holed pair list is undetectable by anything
downstream today, while a chain-form deck cannot express the hole at all
(defense in depth); a human auditing an apeGmsh-generated deck reads
`-master-chain 101 102 103 104` correctly without knowing the stride-2
trap; every hand-written fork deck gets the same protection; and the T4
review's disclosed double-count edge case, "reachable only by compounding
an already-discouraged holed master declaration"
(`85_ladruno_contact_2d_adr.md`, T4 log), becomes unreachable from
chain-form decks.

**Cost, read from the code.** Trivial. `ladrunoContactSurfaceImpl`
(`OpenSeesOutputCommands.cpp:351-497`) already reads a kind token, an
arity, and a flat tag list; a chain kind reads k+1 tags and writes 2k
into the ID before the existing node-existence and dimension-consistency
checks run unchanged. Parser-only, ~30 lines, no handler or kernel
change, no 3-D surface touched.

**If the answer is no.** Nothing blocks. apeGmsh's S2 gate (a
three-segment master emits six tags in chained order; a test asserts the
holed four-tag form is never produced) is fully adequate for apeGmsh's
own decks, which is the adoption's actual requirement.

### Ask 3 — contact-force observability worth a Results layer

**The problem.** The lane's force reporting is thin in three independent
ways. `ladrunoContactForce` is NTS-only — fed exclusively from the
SEGMENT/end-cap branch (`LadrunoContactFE.cpp:1531-1537`); the mortar
lane, the lane with the machine-precision patch-transfer headline, has no
force query at all (only the geometric `ladrunoMortarPenetration`), and
the rigid-plane lane has none either. It reports a **scalar magnitude**,
the per-pair `tn` summed over a slave node's pairs
(`LadrunoContactDomain.h:664`, `map<PairKey, double>`;
`LadrunoContactDomain.cpp:1080-1083`) — the fork's own guide warns this
cannot be read as any global force component near a corner or end-cap,
where the normal is not axis-aligned. And the 3-D twin of the T4
stale-released-pair defect is a known, deferred wrongness: a released 3-D
pair reports its last-active force forever, reproduced at
`f_query = 1000.0` vs `f_true = 0.0` (`LEDGER_quirks.md:4551`; the 2-D
fix is `LadrunoContactFE.cpp:1449-1470`). apeGmsh's Results layer
consumes nothing of this today, so the ask is prospective — but a 2-D
contact adoption whose results story is "one scalar, one lane, and a
known-stale 3-D twin" undersells the lane the fork just spent T0–T4
building.

**The proposed change.** Three parts, separable, in value order: (a) fix
the deferred 3-D stale-release defect — the fork already owes itself this
follow-up, including the deliberate `contact_dump` re-baseline its ledger
row plans for; (b) widen the per-pair snapshot from scalar `tn` to the
force vector — the FE has the normal and the friction traction in hand at
every `setNtsForce` call site — behind a vector-returning query, keeping
the scalar query for compatibility; (c) per-slave-node force accumulation
on the mortar and rigid-plane residual paths, behind the same query.

**Why fork-side beats an apeGmsh workaround.** The numbers only exist
inside the residual assembly; nothing apeGmsh can do recovers a mortar
nodal contact force from the outside short of differencing element
internal forces across the interface, which is exactly the error-prone
post-processing a contact query exists to replace.

**Cost, read from the code.** Medium, and honestly the least favorable
ratio of the three — which is why it is ranked last. (a) is small but
carries the disclosed dump re-baseline. (b) is a contained type change
(map value struct, a handful of call sites, one query). (c) is real new
accumulation code on two more lanes. All of it is query-side only — no
physics, no tangent, no shipped-deck behavior change beyond the
deliberate (a).

**If the answer is no** (or only (a) is taken, which alone is worth
filing): apeGmsh captures NTS magnitudes per step by injecting
`ladrunoContactForce` calls into the generated analysis loop, documents
that mortar and rigid-plane forces are recoverable only via reactions or
the penalty-depth identity, and the Results layer ships without contact
force fields. No adoption slice blocks on this.

### Considered and dropped

**Relaxing `ndf == ndm` on NTS/mortar/tie.** Evaluated against the T0
review note and dropped as genuinely immovable: `FE_Element::setID()`
packs each node's *full* DOF_Group sequentially, so any extra DOF on a
multi-node contact FE shifts every later node's equation slots — silent
mis-assembly, the exact incident class the guard was built from
(`85_ladruno_contact_2d_adr.md` §Why, §How/8). The relaxation would be a
rework of core FE_Element ID packing, not a guard edit. The right layer
for mixed-ndf 2-D models is apeGmsh's own S0: once the node emitter stops
padding a third coordinate, per-node `-ndf` works and contact-surface
nodes can simply *be* ndf 2. Not asked.

**Making the `node` command refuse unconsumed trailing tokens** (the
`-ndf`-swallow of brief §3). A fork-side guard would convert a silent
wrong into a loud one for hand-written decks — but it lives in vanilla
node parsing on the hottest path of every deck, and a strict refusal
would *break every existing apeGmsh 2-D deck*, all of which emit the
padded third coordinate today (`opensees/_internal/build.py:952`). The
failure is ours to stop causing, not the fork's to start refusing. S0
removes apeGmsh's exposure entirely. Not asked; mentioned to the fork
team as a note only if they independently want a warning there.

## B. Revised adoption plan (supersedes §5 of the brief)

The fork asks are filed first — call that **F0 (fork-side)** — but the
plan does not serialize behind them. Everything through S2, plus the base
policy of S3, is unblocked today; only three well-marked items wait on an
answer. Each apeGmsh slice keeps the fork's gate idiom: the existing 3-D
contact battery green with an unchanged pass count, recorded in the PR
body. Nothing in the 3-D lane may move. 2-D under partitioning is refused
by name throughout (out of scope fork-side).

| Slice | Side | Waits on | Content |
|---|---|---|---|
| F0 | fork | — | File asks 1–3. |
| S0 | apeGmsh | nothing | `ndm` coordinates in 2-D decks (`build.py:952`). Gate unchanged from the brief: a mixed-ndf 2-D deck reports the ndf it was told to carry via `llength [nodeDisp $tag]`; 3-D decks byte-identical. |
| S1 | apeGmsh | nothing | Rigid plane: 2-vector acceptance in `contact_plane_args` (`contact.py:264`), zero-padding to the permanently valid 9-arg form, plus a test. **No probe** — see §C.1. Independent of S0 (`ndf ≥ ndm` holds on this lane). |
| S2 | apeGmsh | nothing | The grammar and the pair list: `nps = 2` through `_check_nps` (`contact.py:61`), the corner map (`ConstraintsComposite.py:129` — **not** extensible by key: `_drop_to_corner_facets` keys purely on facet width (`:140`), and a 3-node line3 edge is indistinguishable from a 3-node tri3 facet, so adding `3: 2` would silently remap every 3-D triangle master. The extractor needs the facet's dimension threaded in alongside its width; a `{2: 2, 3: 2}` line table selected by dimension, not merged into the surface table), the H5 schema (`_femdata_h5_io.py:1901`, `:3656` — and the 3-slot outward payload, §C.2); 2-component `-outward` through emit (`contact.py:232`); ordered edge chaining **and winding** derived from solid-element adjacency. The chaining/winding work is common to every fork-answer outcome — the chain-integrity scan FATALs on permuted or reversed listings regardless, and winding feeds S3 if ask 1 lands. Gate: three segments emit six tags in chained order; the holed four-tag form is asserted never produced. If ask 2 is granted, switching emit to the chain form is a mechanical follow-up, not a redesign. |
| S3 | both | **ask 1** (policy only) | NTS with friction, and the orientation policy. Base implementation, correct under every fork answer: flush interfaces raise a named apeGmsh error requiring `outward=` (naming both surfaces, showing the call to add); strongly curved masters refused with split guidance. If ask 1 is granted: emit `-outward winding` from the S2 adjacency-derived winding, and the flush/curved refusals disappear. Either way the 3-D "never auto-derive" comment (`ConstraintsComposite.py:522`) gets its explicit 2-D carve-out — under winding, note that apeGmsh derives no *vector*; the side is declared through chain orientation. |
| S4 | apeGmsh | S2, S3 | Mortar, tie, `-thickness` (parameter currently absent from `contact.py`). The two thickness conventions and the `-epsN auto` no-h-scaling rule as in the brief §5 S4 — all verified still accurate against the fork guide. Mortar shares S3's orientation outcome. |
| S5 | apeGmsh | S3 | Docs: `guide_constraints.md` 2-D section; the curved-master facet-length warning where contact meshing guidance lives; the chained-declaration convention. |
| S6 | apeGmsh | **ask 3** (shape only) | Results-layer contact observability — new slice, absent from the brief. If ask 3 lands: consume the vector query per lane. If not: per-step `ladrunoContactForce` capture injected into the generated analysis loop, NTS magnitudes only, gap documented. Either way, do not trust 3-D `ladrunoContactForce` readings on decks with release events until the fork's deferred stale-fix lands (`LEDGER_quirks.md:4551`). |

Dependency order: S0 first (it is the prerequisite for the `ndf == ndm`
guard to be satisfiable at all); S1 in parallel at any time; S2 → S3 →
S4 → S5; S6 after S3. Unblocked today: F0, S0, S1, S2, S3-base. Waiting
on a fork answer: the S3 winding layer (ask 1), the S2 emit-form switch
(ask 2, cosmetic), and the S6 shape (ask 3). The recommendation is to
proceed through S3-base without waiting — the base policy is the
fallback anyway and is forward-compatible with every grant.

## C. Correctness pass on the existing brief

The brief's load-bearing claims were re-verified against current source
and hold: the three emit-grammar gates (`contact.py:61`, `:232`, `:264`),
the corner-facet map (`ConstraintsComposite.py:129`), the never-auto-derive
policy comment (`:522`), the node emitter (`build.py:952`), the H5 nps
validation (`_femdata_h5_io.py:1901`, `:3656`), the chained-pair trap and
its measured `[5/18, 8/18, 5/18]·P` consequence, the flush-abort and
2-component `-outward` behavior, the thickness conventions including the
`-epsN auto` h²-error rule, and the `ndf == ndm` packing rationale. Three
items need correction or are now obsolete:

1. **§6 Q1 ("Does `contact_plane` already work in 2-D?") is resolved, and
   §2's "may already work by accident" undersells it.** The fork's T0
   shipped and *gated* the rigid-plane lane in 2-D: G-T0 includes
   2-D block-on-rigid-plane statics with SOFT and visc variants, and the
   9-arg zero-padded `contactPlane` on a 2-D surface as an explicit
   back-compat gate (`85_ladruno_contact_2d_adr.md`, G-T0(b)/(c) and the
   T0 log, PR #749, "5 rigid-plane tests"). It is a declared, tested
   capability, not an accident. S1 needs no probe deck — it is a
   2-vector `contact_plane_args` plus an apeGmsh-side test, full stop.
2. **§2's refusal table misses one site.** The H5 payload stores
   `outward` as a fixed 3-slot field on write and read
   (`_femdata_h5_io.py:1930-1936`, `:3679-3682`), so a 2-component
   outward does not round-trip even after the nps validation is widened.
   Small, but it belongs in S2's schema work by name, and the brief's
   table and S2 slice name only the nps validation.
3. **§4's flush-interface framing is superseded, and its curved-master
   hazard is incomplete.** The brief poses the flush policy as a binary —
   require `outward=` or derive a vector from geometry — because under
   the apeGmsh-only constraint those were the options. With fork changes
   possible there is a third option that dominates both: a
   declared-winding orientation mode (ask 1), which requires no vector
   from anyone and is exact per segment. Separately, the brief's §4
   curved-master item covers only the facet-length/penetration disarm
   hazard; it does not mention that the orientation vote **splits and
   aborts on strongly curved masters even with `-outward`**, and that
   closed-loop masters cannot be oriented at all
   (`LadrunoContactHandler.cpp:510-519`; guide, "Flush interfaces require
   `-outward`" section). That refusal is the stronger practical limit on
   curved-master decks and is what ask 1 removes.

Beyond these, one addition rather than a correction: the brief has no
results story. `ladrunoContactForce` being NTS-only, scalar, and
stale-on-release in 3-D (§A ask 3) was invisible to a brief scoped to
"make the deck emit"; an adoption that includes the Results layer needs
the S6 slice above.

## See also

- `contact_2d_adoption.md` — the original brief; §§1–4 remain the
  adoption map, §5 is superseded by §B here, §6 partially by §C.
- `OpenSees/Ladruno_implementation/85_ladruno_contact_2d_adr.md` §How/2
  and §How/8 — the orientation vote and the guard contract the asks are
  argued against.
- `OpenSees/Ladruno_implementation/LadrunoContact2D_guide.md` — kernel
  behaviour; the authority once decks run.
