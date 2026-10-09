### CHANGED — the `viewer` and `all` extras pin `PySide6<6.12` (#1571)

PySide6 6.12.0 deletes every live Python-wrapped widget at interpreter exit
(`destroyQCoreApplication`). A Qt viewer panel kept alive by its own lambda
slots after it has been dropped is then deleted twice, and the process
segfaults on the way out, after the script's work is done. This was
reproduced with the model viewer's Boolean panel on Linux. The Qt viewers
are in sunset (ADR 0112) and are removed at V5, so the extras pin
`PySide6>=6.5,<6.12` until then rather than reworking the panels. Every
viewer test module now deletes the top-level widgets it creates, so the
suite no longer leaves such widgets alive at exit and passes under 6.12.
