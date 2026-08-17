# tools/ — intent bridges for this habitat

Launchers and sketch archives for **apeCAD** and **apeSketch**. Part of the
APE Studio template: copy with the habitat; keep FEM under `models/`.

```text
tools/
├── README.md                 ← you are here
├── _launch.py                ← shared path / PYTHONPATH helpers
├── apeCAD/
│   ├── open_interface.py     ← open apeCAD scratchpad
│   ├── human_sketches/       ← human-authored CAD docs / images
│   └── agentic_sketches/     ← agent-authored CAD docs / images
└── apeSketch/
    ├── open_interface.py     ← open apeSketch board
    ├── human_sketches/       ← human ink sessions / exports
    └── agentic_sketches/     ← agent schemes / exports
```

## Open the interfaces

From the habitat root (uses `opensees_env` if on `PATH`, else `python`):

```text
python tools/apeCAD/open_interface.py
python tools/apeSketch/open_interface.py
```

Optional:

```text
python tools/apeCAD/open_interface.py --role human
python tools/apeCAD/open_interface.py --role agentic --no-browser
python tools/apeSketch/open_interface.py --role human
python tools/apeSketch/open_interface.py --role agentic --http-only
```

`--role` selects which sketch folder is the default save / session root
for that run (`human_sketches` vs `agentic_sketches`).

Clones are expected under `<github>/apeCAD` and `<github>/apeSketch`
(`~/Documents/Github/…`) unless `APECAD_SRC` / `APESKETCH_SRC` point at
`…/src`.

## What goes where

| Folder | Who | Content |
|--------|-----|---------|
| `*/human_sketches/` | Human | Ink boards, CAD JSON, screenshots confirming intent |
| `*/agentic_sketches/` | Agent | Restatement schemes, labeled drafts for Socratic confirm |

Use these archives in the Socratic loop (`APE/instructions/socratic-geometry.md`).
They are **not** FEM models — realization stays in `models/<id>/src/`.
