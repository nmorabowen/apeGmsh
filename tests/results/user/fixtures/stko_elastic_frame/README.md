# stko_elastic_frame

`results.mpco` and `results.mpco.cdata` are byte-for-byte copies of
`stko_results_examples/elasticFrame/results/` from STKO_to_python
(github.com/nmorabowen/STKO_to_python, MIT License, © 2025 Nicolas Mora Bowen;
the files were last changed in commit `18a82ed`). The fixture is a portal frame of three
`ElasticBeam3d` elements over ten steps, as STKO's MPCO recorder wrote it.
It is the cut-kernel oracle for `/entities` (VU8 #2692):
`test_user_entities.py` and `make_app_entities_fixtures.py` read it.
