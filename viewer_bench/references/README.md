# references/ — optional project literature

Place **PDFs, papers, design notes, drawings exports, and other
external material** that inform *this* habitat’s FEM model here — when
it is useful. This folder is **not required**.

```text
references/
├── README.md          ← you are here
└── (optional files)   ← e.g. papers/, codes/, drawings/, notes.md
```

## Soft rule

| Situation | Action |
|-----------|--------|
| Paper, code excerpt, or hand calc that the model must match | Add it (or a stable link + citation in `notes.md`) |
| Generic textbook everyone already knows | Skip — cite in `APE/memory/` instead |
| Huge binary only used once | Prefer link / local archive outside git; or git-ignore the heavy file |
| Nothing yet | Leave the folder empty aside from this README |

## How it relates to the rest of the habitat

| Location | Role |
|----------|------|
| `references/` | External sources for *this* model (optional) |
| `APE/memory/` | What we decided and assume (durable project memory) |
| `APE/instructions/` | How we work across projects (template) |
| `reports/` | Authored narrative and promoted figures |
| `postmortem/` | Session metrics and friction backlog |

If a reference defines an **oracle** (expected stiffness, capacity, test
curve), point to it from `APE/memory/brief.md` or `assumptions.md` and
from the progressive-modeling checkpoint — do not rely on chat alone.

## Agent guidance

- MAY read files here when the task cites them or the brief points here.
- Do not treat an empty `references/` as a missing requirement.
- Prefer portable citations (`Author, Year` + path or URL) over pasting
  long PDF text into chat.
