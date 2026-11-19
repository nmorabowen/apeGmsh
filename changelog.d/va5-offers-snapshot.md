### apeGmshViewer: `--present` offers and `--snapshot` (ADR 0128 slice 5)

- `--mode=present --spec=<spec.json>` offers a view to the running window, one offer per shot. The window lists it and changes nothing else; the view moves only when the human clicks Apply or Back. With no window running it exits 1 and opens nothing; a refused spec exits 2.
- `--mode=snapshot --out=<dir>` reads the human's view back: `view.json` (a spec), `view.png` and a `manifest.json` entry with the selection, and it names what a v1 spec cannot say about the view.
- A collapsed right rail shows the count of pending offers on its toggle.
- Fixed: a section plane given with a non-unit normal now cuts where the spec says, `normal . x = offset` with the normal as given. `normal [2,0,0], offset 4` cuts at x = 2, in `--render` stills too; it cut at x = 4 before.
