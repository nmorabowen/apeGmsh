### CHANGED — elastic-beam line stations: solver oracle on MPCO and .ladruno, and `RecorderTranscoder` needs a model (#1956, G17)

An elastic beam's line stations, the two-station slab built from `localForce`,
were checked only for shape and self-consistency on MPCO. The new fork-lane
test `tests/opensees/integration_ladruno/test_elastic_beam_stations_live.py`
solves a 3-D `elasticBeamColumn` cantilever and a 2-D simply supported beam
through the bridge, and compares every station with `localForce` and with
the closed form (`P L` at the root, `P L / 4` at midspan) at `rtol=1e-10`. The
expected values follow the force-beam section convention, so an elastic beam
meets an adjacent force beam without a jump. The components that disagree
today are strict xfails on #2009 (sign), #2010 (MPCO 2-D shear and moment)
and #2011 (`basicForce`-only `.ladruno`).

`tests/test_results_line_station_lineage_refusal.py` proves the elastic-beam
check refuses a `model=` built from another FEMData, called directly and
through `RecorderTranscoder`. Before it, removing that refusal passed every
test.

`RecorderTranscoder` now refuses an elastic-beam line-stations record when no
`model_h5_src` is given, naming the record. Without a model it cannot check that
every element is an elastic beam: a force beam in the record would read as two
end stations instead of its sections. `Results.from_recorders` always passes a
model, so only direct `RecorderTranscoder` calls are affected.
