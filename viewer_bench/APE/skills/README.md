# Skills — references

Portable catalog of agent skills for this APE Project. Skills are **not**
auto-loaded; agents **Read** a `SKILL.md` when the task matches.

| File | Purpose |
|------|---------|
| `catalog.md` | Human door — domains + recommended first reads (empty until harvested) |
| `catalog.json` | Machine twin from harvest (empty until harvested) |
| `harvest.py` | Re-scan GitHub-canonical (+ legacy) skill trees on **your** machine |

```text
python APE/skills/harvest.py
```

Policy: prefer `<github>/…/SKILL.md` (`~/Documents/Github/`). See
`APE/instructions/how-we-work.md`.

## No skill catalog ships with this template

`catalog.md` / `catalog.json` start empty. Any recommended skill stack
(engineering-domain SKILL.md doors such as `abaqus-theory`, `aci-concrete`,
`opensees-expert` / `opensees-performance`, `apegmsh`, and similar) is the
**template author's own personal set** — it is not distributed or required
by this template. Run `harvest.py` to build a catalog from whatever
`SKILL.md` files exist in **your** environment.

**Agent guidance:** if any of these skills happen to be installed in your
environment, use them; if not, ignore these references and proceed — they
are never required.
