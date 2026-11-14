### FIXED — viewer manifest test knows `variants.path-particles-sibling.json`, so `main` is green again (#2994)

The invalid-example table in `apeGmshViewer/test/variants/manifest.test.ts`
had no row for the example #2989 added; the test now expects `bad-path` at
row 1 for it. No runtime change.
