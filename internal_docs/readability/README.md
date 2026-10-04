# Readability workshop: "apeGmsh Script English"

This folder holds a draft standard for the **model scripts** that agents write with apeGmsh. It
does not cover library code. The standard asks that a script read like a clear engineering
procedure to a structural engineer opening it cold, with no loss of modelling power.

| File | Holds |
|---|---|
| [REPORT.md](REPORT.md) | the assessment: did the contract work, what it cost, per-model findings, rules that earned their place, API gaps, and the next steps (§6) |
| [contract_v0.md](contract_v0.md) … [contract_v3.md](contract_v3.md) | the contract as it evolved over the three loops; v3 is the latest and has never been run |
| [api_gaps.md](api_gaps.md) | API gaps that forced unreadable code, ranked |
| [HANDOFF.md](HANDOFF.md) | the session handoff from 2026-10-04, kept as written apart from the paths |
| `samples/` | the four baseline scripts written before the workshop |
| `loop1/` … `loop3/` | the scripts from each loop (Pratt truss, two-storey frame, staged strip footing), one file per writer slot a–f |

The run outputs (HDF5, MPCO, CSV, logs, plots) are not kept. The silent hazards the workshop
found are filed as issues #1332–#1338 (and #1259). Its sample-driven fixes are #1321–#1328.
