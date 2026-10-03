### FIXED — invalid escape sequences in two docstrings no longer raise SyntaxWarning (program slice B, #1298)

The `stage_marker_name` docstring in `opensees/_internal/build.py` and `LayerSectionMetadata` builder docstring in `results/capture/spec.py` are now raw strings. Runtime values are unchanged; importing apeGmsh is warning-free on Python 3.12.
