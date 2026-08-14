"""PoC script for apeGmsh.studio (ADR 0095).

Replay without solving::

    python -m apeGmsh.studio tests/studio/fixtures/box.py --phase model --no-viewer
    python -m apeGmsh.studio --status

``--phase mesh`` meshes and stops before ``apeSees``.
"""

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="studio_box") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.from_label("body", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
apeSees(fem)
