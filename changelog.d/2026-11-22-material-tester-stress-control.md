### ADDED — the material tester runs stress control

The tester's form in the Materials & sections window gains **stress control** (ADR 0135 D6, S3): give
target stresses and steps per leg, and a fixed driver in the fork's tester finds the strain of each
step, committing only the strain it chose. Its search watches for the response turning back, so it
reaches a target just below a softening material's peak (Concrete02 to -29 with its peak at -30,
which a doubling search misses), and a target beyond the peak stops the run at that step. The
curve keeps the steps reached, and the run says how far the material got. The imposed-history plot
shows the stress for these runs.
