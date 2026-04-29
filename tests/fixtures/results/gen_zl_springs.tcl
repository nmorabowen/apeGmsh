# Generate a small MPCO file containing ZeroLength elements to validate
# the layout assumptions in _mpco_spring_io.py.
#
#   - elem 100 : 1 spring  (-dir 1 -mat 1)         -> 1-spring bucket
#   - elem 200 : 3 springs (-dir 1 2 3 -mat 1 1 1) -> 3-spring bucket
#
# Ramped SP constraints drive the springs so force / deformation are
# nonzero across multiple steps.

wipe

model basic -ndm 3 -ndf 3

# Pair A (elem 100): coincident nodes 1, 2 at origin
node 1 0.0 0.0 0.0
node 2 0.0 0.0 0.0
# Pair B (elem 200): coincident nodes 3, 4 offset to 1.0 0 0
node 3 1.0 0.0 0.0
node 4 1.0 0.0 0.0

uniaxialMaterial Elastic 1 1000.0

fix 1 1 1 1
fix 3 1 1 1
# Free node of pair A (only x has stiffness; clamp y, z)
fix 2 0 1 1
# Free node of pair B is free in all 3 dofs

# 1-spring zero-length
element zeroLength 100  1 2  -mat 1   -dir 1
# 3-spring zero-length
element zeroLength 200  3 4  -mat 1 1 1 -dir 1 2 3

# Ramp nodal forces (drives the springs)
timeSeries Linear 1
pattern Plain 1 1 {
    load 2 200.0   0.0   0.0
    load 4 100.0  50.0  25.0
}

# MPCO recorder for force + deformation
recorder mpco "zl_springs.mpco" -N "displacement" -E "basicForce" "deformation"

constraints Transformation
numberer Plain
system BandGeneral
test NormDispIncr 1.0e-8 25 0
algorithm Linear
integrator LoadControl 0.2
analysis Static

for {set i 0} {$i < 5} {incr i} {
    set ok [analyze 1]
    if {$ok != 0} {
        puts "analyze failed at step $i"
        exit 1
    }
}

wipe
puts "Wrote MPCO -> zl_springs.mpco"
