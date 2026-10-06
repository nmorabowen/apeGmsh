### CHANGED — Contact tags and partitioned contact routing come from the build-time tag plan (K1-3d S3b, #1479)

The contact allocation loop moved out of `emit_contacts` and
`emit_contact_planes` into `plan_contacts`, which `plan_tags` runs once per
emit mode. On a partitioned emit, `plan_tags` also resolves each
interaction's owner rank and ghost nodes once
(`BuiltModel._plan_partitioned_contacts`), and numbers the interactions rank
by rank, as the deck always did. The emit reads the tags and the routing
from the plan. The routing's partial-ownership warning still fires once per
emit. The emit fork now freezes `contactSurface` and `contact`, so an
emit-time contact mint raises `TagLawError`. Both helpers still accept a
plain `TagAllocator` from a direct caller until K1-3d S6, and drop their
unused `records=` parameter. A helper handed a fork of another model's plan
now says so, for transforms too, and the emit refuses a contact plan that
does not hold each contact record exactly once. Every emitted deck,
partitioned contact decks over 2 and 4 ranks included, is byte-identical.
One ordering change, which the maintainer accepted: the partitioned contact
routing refusals (undecidable owner, a cut master with `kn='auto'`, a
partially traced master under auto sizing, and a contact on a STAGED
partitioned model) now raise when the emit plans its tags. That is before
the partitioned path's own refusals, which now come second: the
`equation_constraint` (EQ) rows refusal, the reinforce, rebar and embed
refusals, the `soft=` refusal, the node-pair ZeroLength refusal, and the
staged BC validators. Every one of them is still a `BridgeError`, so a
partitioned model that breaks more than one rule may now report a different
first error. The pattern-`sp` check on contact ghosts still runs in the
emit.
