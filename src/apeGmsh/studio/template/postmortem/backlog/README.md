# Friction backlog — how it becomes work

Session `02_friction.md` files **diagnose**. This folder **queues**.
Improvement surfaces: `APE/instructions/continuous-improvement.md`.

## Flow

```text
session closes
    → 01_metrics.md + 02_friction.md
    → triage rows into backlog/open.md
    → (optional) open upstream issue/ADR in
         OpenSees/Ladruno, apeGmsh, apeSees, Studio, …
    → NEXT session starts by reviewing P0/P1 here
```

## Priority

| P | Meaning |
|---|--------|
| P0 | Blocks habitat / repeated token hemorrhage |
| P1 | Next habitat week |
| P2 | Backlog when touching that library |
| P3 | Nice / research |

## Rules

1. Every open row needs a **target** tag, a **concrete proposal**, and a
   **next-postmortem proof** (how we will know it worked).
2. Prefer the **smallest wedge** that would have prevented the friction.
3. Instruction-only fixes go to `APE/instructions/` (`ape-instructions`);
   still list them here for visibility.
4. Do not duplicate: merge when the same pain returns; bump evidence links.
5. Closed rows move to “Done” with date + link (commit, ADR, issue, or
   instruction path).
6. Skills and library code are two surfaces among many — also triage
   Studio, observability, oracles, continuity, human-loop, repro, evals.
