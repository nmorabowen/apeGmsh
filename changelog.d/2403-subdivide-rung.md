### ADDED — `ops.strategy.Subdivide(n)`: a transient ladder rung that retries a failed step in `n` pieces (#2403, ADR 0057 Amendment 1)

A staged transient deck could not cut its time step when an increment
failed. The ladder held algorithm rungs only, and `Substep` is a
load-stage dλ policy that decks refuse. A 1-D site-response shaking
stage therefore aborted at its first hard step.

A `Subdivide(n)` rung in `s.run(..., dt=, strategy=ops.strategy.Ladder(
rungs=(...)))` re-issues rung 0's algorithm and runs `analyze(1, h)` up
to `n` times, where `h` is the increment's remaining time divided by `n`.
A failed piece hands the remaining time to the next rung, so the
increment always ends exactly on the step grid and model time never
drifts. Exhausting every rung still aborts the deck: no step is skipped.

- The py and tcl emitters write the loop. Decks without the rung are
  byte-identical to before.
- A static stage refuses the rung at `s.run`.
- A flat live `ops.analyze(strategy=)` cannot run it yet: the live
  emitter does not know the loop, and OpenSees rejects the rung's
  reserved token loudly.
- Ladders are not archived, so there is no schema change.
- A recorder writes one row per committed piece. Read by time when the
  rung fires.
