### TESTS - apeGmshViewer: `npm run typecheck` and `npm run lint` now cover `scripts/**` (#2206)

`tsconfig.scripts.json` typechecks `scripts/**/*.ts` and `depcruise` runs over `src scripts` with a new `no-apegmsh-import` rule (ADR 0112 D4), so `scripts/pair_oracle.ts` can no longer drift from `pairResults` unnoticed.
