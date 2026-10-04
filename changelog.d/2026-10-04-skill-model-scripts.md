### ADDED: Script English reference scripts in the skill

The skill gains `references/model-scripts.md`: a 10-rule card for agent-written model scripts
(shape, labels not tags, named numbers with units, hand checks that are asserted), plus three
runnable reference scripts: a 2-D Pratt truss, a 3-D frame with rigid floors and a modal
analysis, and a staged strip footing on J2 clay. `tests/test_skill_model_scripts.py` runs them
in the `live` lane. `gotchas.md` warns that linear quads with J2 soil lock, and that `analyze`
can return 0 on a singular stiffness. The cheatsheet signatures of `set_transfinite_curve`
and `rigid_diaphragm` are corrected.
