# ADR 0036 — Embedded-host decomposition: linear coupling over corner nodes

**Status:** Accepted.  No OpenSees changes; apeGmsh-side only.  No
H5 schema bump (`EmbeddedDef` is a pre-mesh declaration, never
serialised).  Complements ADR 0022 (the MP-constraint fan-out that
makes `ASDEmbeddedNodeElement` the single OpenSees primitive every
embed lowers to) and ADR 0035 (which exposed the element's C++
optionals).

## Context

`element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 $Rnode2 $Rnode3
<$Rnode4> …` accepts **exactly 3 or 4 retained-node tags**
([cpp:201](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)).
The C++ constructor stores the tags without validating their source
([cpp:293-322](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp));
`setDomain` checks only node existence, `ndm`, `ndf`; the stiffness
builders run **linear** tri / tet shape functions over the retained
node coordinates ([cpp:534-585](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)).
The element does not know — and cannot know — whether the 4
retained tags came from a real tet4, 4 of a hex8's 8 corners, or
4 arbitrary points.

Before this ADR, `_collect_host_elems` (the apeGmsh-side host
gatherer) **only accepted tri3 (etype 2) and tet4 (etype 4)** host
elements.  A quad-recombined shell host (etype 3), a brick host
(etype 5), or any higher-order host (tri6, tet10, quad8/9, hex20)
raised a fail-loud error at constraint-resolution time, forcing the
user to remesh the host with simplex algorithms before they could
embed anything.

This was overly conservative.  Users who model a concrete brick
with hex8 (a structural-FEM standard) had to either:

1. Re-mesh as tet4 (poorer element conditioning, more elements,
   user-visible change to the host's physics), or
2. Hand-build the embedded constraint by writing
   `element ASDEmbeddedNodeElement $tag $cnode <4 of the 8 hex
   corners> -K 1e8` directly in Tcl/Python — exactly what apeGmsh's
   `g.constraints.embedded(...)` exists to abstract.

The user's hand-built workaround (option 2) was already in
production and produced numerically correct results, which surfaced
the architectural insight: **the C++ element accepts any 4 tags as
an implicit tet — we can decompose non-simplex hosts on the
apeGmsh side without touching OpenSees**.

## Decision

`ConstraintsComposite._collect_host_subelements` (renamed from
`_collect_host_elems` to make the new contract explicit) decomposes
non-simplex and higher-order hosts into linear sub-tris / sub-tets
using **corner nodes only**, on the fly:

| Gmsh etype       | Code  | Decomposition                                            |
|------------------|-------|----------------------------------------------------------|
| tri3 (CST)       | 2     | identity                                                 |
| tet4             | 4     | identity                                                 |
| quad4            | 3     | 2 tris via the (0,2) diagonal split                      |
| hex8             | 5     | 6 right-handed Kuhn tets (`HEX8_TO_6_TETS`)              |
| prism6           | 6     | 3 tets (`PRISM6_TO_3_TETS`)                              |
| pyramid5         | 7     | 2 tets (`PYRAMID5_TO_2_TETS`)                            |
| tri6 (LST)       | 9     | corners only → 1 tri (midsides discarded)                |
| tet10            | 11    | corners only → 1 tet                                     |
| pyramid13        | 14    | corners only → 2 tets                                    |
| quad8 / quad9    | 16/10 | corners only → 2 tris                                    |
| hex20            | 17    | corners only → 6 Kuhn tets                               |
| prism15          | 18    | corners only → 3 tets                                    |

The returned rows are **virtual** — they do not correspond to
elements in the gmsh mesh.  They exist purely as a coupling-layer
fabrication so the linear-shape-function coupling of
`ASDEmbeddedNodeElement` works against any supported host topology.

### The four mitigations that make this safe

Because `_collect_host_subelements` is now a fabrication function
(not a transparent collector), four invariants are enforced
in-PR:

1. **`host_coupling="linear"` reserved keyword.**  `EmbeddedDef`
   gains a `host_coupling: str = "linear"` field with a
   `__post_init__` guard that rejects any other value.  Reserving
   the keyword now (rather than after the fact) means a future
   `"trilinear"` / `"biquadratic"` option can be added without
   changing the public API; pre-existing models keep producing
   identical numerical results because `"linear"` stays the default.
2. **Kuhn-table orientation property test.**  Every row of
   `HEX8_TO_6_TETS` / `PRISM6_TO_3_TETS` / `PYRAMID5_TO_2_TETS`
   is evaluated against Gmsh's canonical reference coordinates
   fetched live via `gmsh.model.mesh.getElementProperties(etype)`,
   and asserted to have positive signed volume.  If Gmsh ever
   renumbers any host primitive, the test fails immediately
   rather than letting wrong-signed shape functions silently land
   in `ASDEmbeddedNodeElement`.
3. **Mixed-dim host fail-loud.**  When a host PG produces both 2D
   sub-tris and 3D sub-tets in the same collection pass, the
   collector raises with an actionable message — the embedded
   coupling cannot pick between them deterministically (kNN
   centroid search would dispatch based on opaque proximity, which
   is opaque physics).  Split the host PG into two
   `embedded(...)` calls instead.
4. **Higher-order warning.**  When a midside-bearing host (tri6,
   tet10, quad8, quad9, hex20, prism15, pyramid13) hits the
   decomposition path, the collector emits **one `UserWarning`
   per (etype, entity)** pointing at the linear-coupling
   consequence.  The user sees "embedded: host entity carries
   tri6 elements — decomposing to corner-node-only linear
   sub-elements.  The embedded coupling will be linear regardless
   of the host's native interpolation order; quadratic / bilinear
   / trilinear host kinematics will NOT be felt by the embedded
   node."  Acknowledge by setting `host_coupling="linear"`
   explicitly on the `embedded(...)` call (a no-op selection that
   reads as "I understand the coupling order").

### Decomposition choices

**Hex8 → 6 Kuhn tets (not 5).**  The Kuhn-Freudenthal
6-tet decomposition shares the main diagonal (vertex 0 → vertex
6), is symmetric, and is orientation-independent of neighbouring
hex8 decompositions.  The 5-tet decomposition is faster but only
maintains face-conformity when adjacent hexes use the *same*
orientation, which doesn't matter for embed-purposes (we don't
solve a PDE on the sub-tets) but is a footgun if anyone ever
repurposes the sub-tets.  6-tet is the safer default.

**Quad4 → 2 tris on the (0,2) diagonal (not both diagonals).**
For convex quads, the (0,1,2) ∪ (0,2,3) coverage equals the quad.
The resolver's centroid-kNN search already returns K=16 candidates
and picks the lowest-excess; an embedded point inside the quad
lands in at least one sub-tri.  Both-diagonal coverage (4 tris per
quad) would only help bowtie / non-convex quads, which are
already a meshing quality issue.

**Higher-order (tri6, tet10, hex20, etc.) → corner-only.**  The
embedded coupling is linear regardless; midside nodes are not in
the retained set for any reduction we could do without inventing a
new OpenSees element class.  Discarding the midsides is honest
(see B7 warning).

## Alternatives rejected

**Refuse non-simplex hosts (status quo).**  Forces users to
remesh as tet4 / tri3, which changes the host's physics for an
unrelated reason.  Already in production we had users
hand-building the workaround — the abstraction had a hole.

**Per-host projector strategy interface
(`HostProjector`).**  The clean architectural fix: each gmsh
etype has its own projector (hex8 → trilinear projection onto 8
corners; tri6 → quadratic onto 6 nodes; etc.).  Requires:

- A new OpenSees element class that supports N-node retained sets
  with native shape functions (e.g. `ASDEmbeddedHex8` with
  trilinear shape functions over 8 retained nodes).
- Element registration through `classTags.h`, `OPS_*` factory,
  `FEM_ObjectBroker::getNewElement`, `sendSelf` / `recvSelf` for
  parallel runs.
- apeGmsh side: `host_coupling="trilinear"` / `"biquadratic"`
  branches in both the collector (skip decomposition) and the
  emitter (emit the new element class instead of
  `ASDEmbeddedNodeElement`).

This is a 2–3 month project requiring coordinated PRs against
ASDEA's OpenSees and an RFC for the new element class.  The
`host_coupling` reserved keyword is the hook that lets it land
later as a non-breaking change.  We do not block immediate use of
non-simplex hosts on that work.

**Both quad diagonals (4 tris per quad).**  Doubles the candidate
pool for the kNN search.  No benefit for convex quads; only
matters for bowtie quads, which are already a mesh-quality issue.
Add later if a real bowtie-quad case lands.

## Consequences

### Positive

- Users can now embed into hex8 / quad4 / prism6 / pyramid5 hosts
  directly via the existing `g.constraints.embedded(...)` API,
  without remeshing.
- No OpenSees changes required; ships immediately.
- No H5 schema bump (the `host_coupling` field lives on
  `EmbeddedDef` which is pre-mesh-only).
- The hand-built workaround (option 2 above) deletes from user
  code in favour of one library call.

### Negative (acknowledged)

- **Per-hex coupling asymmetry.**  Two embedded nodes inside the
  same hex8 may couple to *different* 4-corner subsets depending
  on which of the 6 Kuhn sub-tets contains each one.  This is
  geometrically correct under linear coupling but can surprise
  readers of the resolved records.  Documented in `EmbeddedDef`
  docstring and in this ADR.
- **Linear-over-corners is lossy for higher-order hosts.**  A
  user models an LST plate for bending and embeds rebar: the
  embed sees only the linear corner stretch, not the quadratic
  curvature that motivated LST in the first place.  Documented
  via the B7 warning fired on first encounter.
- **Sliver sub-tets.**  Badly-shaped (high-aspect-ratio) hexes
  produce sliver sub-tets; the C++ penalty stiffness
  `iK = m_K · √V` becomes small.  The B6 sliver-tet guard test
  documents the current behaviour (resolver still produces
  bounded weights or fail-loud raises on degenerate det).

### Neutral

- `_collect_host_elems` rename to `_collect_host_subelements` is
  technically a breaking change to a private API.  No downstream
  user-code surface affected; the only external reference was a
  test in `tests/test_constraint_emission.py` (updated in-PR).
  ADR 0027 §"Mixed-host silent drop" and `docs/api-flows/*`
  historical references updated in-PR.

## Future work (deferred, not blocking)

- `HostProjector` abstraction + new OpenSees element classes for
  trilinear / biquadratic coupling (see Alternatives).  RFC owed
  before any commitment.
- Both-diagonal quad split for bowtie quads (only if a real case
  surfaces).
- Sliver-tet diagnostic in the resolver (warn when sub-tet
  aspect ratio exceeds a threshold).

## References

- [`ConstraintsComposite._collect_host_subelements`](../../../core/ConstraintsComposite.py)
- [`HEX8_TO_6_TETS` / `PRISM6_TO_3_TETS` / `PYRAMID5_TO_2_TETS`](../../../core/ConstraintsComposite.py)
- [`EmbeddedDef.host_coupling`](../../../_kernel/defs/constraints.py)
- [`tests/test_embedded_decomposition.py`](../../../../../tests/test_embedded_decomposition.py)
- ADR 0022 — MP-constraint emission fan-out
  ([`0022-mp-constraint-emission-fanout.md`](0022-mp-constraint-emission-fanout.md))
- ADR 0035 — ASDEmbeddedNodeElement option exposure
  ([`0035-asd-embedded-node-element-option-exposure.md`](0035-asd-embedded-node-element-option-exposure.md))
- Upstream C++ — [`ASDEmbeddedNodeElement.cpp`](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)
  (original implementation by Massimo Petracca, ASDEA Software
  Technology)
