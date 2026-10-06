### CHANGED — API docs render inherited members for `apeSees`, `ConstraintsComposite` and `Results` (program slice C3.3, #1430)

The three `docs/api` directives (`opensees.md`, `constraints.md`, `results.md`) set `inherited_members: true`, so a mixin split of these classes cannot silently drop public methods from the published API reference. Docs-only; per-directive, no `mkdocs.yml` change.
