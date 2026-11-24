### ADDED — apeGmshViewer tool windows: the tool host and its first tool (ADR 0137 H1)

Tools can now run in windows of their own beside the model window, for example on a second
monitor. Spotlight (Ctrl+K) opens the first, **Open Tool Window: Step Readout**: a separate window
showing the results stage, step and playback, with previous and next step buttons that move the
model window. Each tool window is a separate sandboxed renderer that sees only a small context
from the model window and changes anything only by sending an event the model window checks
first. If a tool window stops or stops responding, the model window keeps running and its banner
says so; opening the tool again starts a fresh window. Under 1 GB of available RAM, tool windows
are not opened and open ones pause until memory frees. The banner now also updates the memory
warning of #3057 on every reading during a load, not only when the load starts or ends.
