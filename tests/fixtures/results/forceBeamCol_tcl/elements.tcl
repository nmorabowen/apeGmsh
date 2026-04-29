
# beam_column_elements forceBeamColumn
# Geometric transformation command
geomTransf Linear 1 1.0 0.0 -0.0
element forceBeamColumn 1 1 2 1 Lobatto 3 5
# Geometric transformation command
geomTransf Linear 2 0.0 0.0 1.0
element forceBeamColumn 2 2 3 2 Lobatto 3 5
# Geometric transformation command
geomTransf Linear 3 -1.0 0.0 0.0
element forceBeamColumn 3 3 4 3 Lobatto 3 5
