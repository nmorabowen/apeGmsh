### CI: raise the suite job timeout from 25 to 40 minutes

The `suite` job in `tests.yml` passed in 1,416 s on main and was then cancelled by its 25-minute clock during teardown, leaving main unverified. The cap is now 40 minutes (#2166).
