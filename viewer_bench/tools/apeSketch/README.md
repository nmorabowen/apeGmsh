# apeSketch — habitat tool door

Open::

```text
python tools/apeSketch/open_interface.py
python tools/apeSketch/open_interface.py --role agentic
```

| Folder | Purpose |
|--------|---------|
| `human_sketches/` | Human ink sessions (`sessions/`, `assets/`) and exports |
| `agentic_sketches/` | Agent schemes / session exports for Socratic review |

`--role` redirects the Session store into that folder so sketches stay
inside this habitat (not only under the apeSketch clone). See `tools/README.md`.
