# To-review notebooks

These notebooks were not part of the EOS recorder/viewer wiring sweep
in PR (this branch). Each falls into one of three categories:

## Pre-existing API drift not fully recovered

| Notebook | Reason |
|---|---|
| `example_column_nodeToSurface_v6.ipynb` | Uses the removed `Results(node_coords=..., cells=..., cell_types=...)` direct constructor for an unstructured-grid augmented visualization. Needs a port to current `NativeWriter` + custom mesh writing. |
| `example_LTB_shell.ipynb` | Spawns a long-running OpenSees subprocess (>120 s); the cp1252 encoding bug is fixed in this branch but the subprocess timeout remains. |
| `example_plate_pyGmsh.ipynb` | `fem.elements.connectivity` on a mixed-element-type mesh now raises (was permissive in the old broker). Needs `.resolve(element_type='tri3')` adoption throughout. |
| `example_plate_viewer_v2.ipynb` | Deeply chained on `Results.to_vtu` / `.to_pvd` / `.to_mesh_data` / `ProbeEngine` — all removed. Model build + DomainCapture wiring works; legacy post-processing chain needs a tutorial rewrite. |

## Structurally incompatible with single-FEM DomainCapture

| Notebook | Reason |
|---|---|
| `09_mesh_refinement.ipynb` | Function-wrapped sweep that rebuilds the mesh (different `fem`) on each iteration. |
| `20_time_history.ipynb` | SDOF mass-spring built directly with `ops.node(1, ...)` — no FEMData broker. |
| `21_opensesmp_block_2parts.ipynb` | Parallel/MPI specific; build code does not bind a `fem` variable. |
| `example_frame3D_slab_opensees.ipynb` | Builds the OpenSees model without going through the FEMData broker. |
| `example_frame3D_slab_opensees_manual_results.ipynb` | Same. |
| `example_plate_basic.ipynb` | Multi-stage flow with five `ops.wipe()` calls; build code does not match the standard pattern. |

## Physics limitation acknowledged

| Notebook | Reason |
|---|---|
| `example_LTB_swept_solid.ipynb` | LTB requires geometric nonlinearity that solid elements in OpenSees do not support. The captured analysis cannot reach a converged buckling state. |
| `example_LTB_swept_solid_v2.ipynb` | Same. |
| `example_LTB_swept_solid_v3_twist.ipynb` | Same. |

---

To revisit one: move it back to `examples/EOS Examples/`, then either re-run
the auto-wiring (`_wire_eos.py`) or hand-rewrite the post-processing path
to use the current API.
