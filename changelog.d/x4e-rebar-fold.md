### CHANGED — `g.rebar` lives in `apeGmsh.rebar` (program slice X4-e, #1809)

The `g.rebar` composite moved from `apeGmsh.core.RebarComposite` to `apeGmsh.rebar._composite`, next to
the detailing standards and hook geometry it uses. The public surface is unchanged: `g.rebar.*` and
`from apeGmsh.rebar import ...` work as before. `apeGmsh.core.RebarComposite` stays as a re-export facade
until 2027-01-11 (retirement-ledger row `facade-core-rebar-composite`). The move also trims narrative
comments and folds the duplicated "designation without a standard" error path (same message), so the
rebar ratchet group shrinks from 2015 to 1924 lines. `tests/rebar/test_rebar_public_surface.py` pins all
62 public rebar symbols and their parameter names.
