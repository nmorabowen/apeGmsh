### ADDED — apeGmshViewer: a Springs card with force arrows on zero-length springs

When the results stage on screen is a `.ladruno` that records `force` for zero-length springs (2-node elements whose
two nodes coincide, such as `zeroLength` and the springs `g.constraints.interface()` emits), a **Springs** card
appears in the left rail. Its **3D spring forces** switch (off by default) draws one arrow per spring: the tip on the
spring, along the force node j applies to the spring (tension-positive, as the Qt diagram's basic force: a spring
stretched along +x draws +x), its length set by the magnitude against the step's largest. The force is read in the
global frame (the `force` token's `P2_1..P2_ndm` columns), so a skewed (`-orient`) or an interface spring is drawn
exactly, with no spring frame to recover; at a spring's ground node the Reactions card draws the opposite arrow. A
failed read of the step's springs says why on the card. A menu draws the **resultant**, its **X, Y, Z
components** along the global axes, or one of them. The card has its own scale, head size (pixels), colour (one
colour, or jet over the magnitude) and thinning, which keeps or drops a spring's components together, and leaves
out forces under 1e-6 of the step's largest. Spotlight offers the toggle. The Qt `SpringForceDiagram` draws one
spring direction at a time (`spring_force_<n>`); per-direction arrows from `basicForce` are a follow-up. A
partitioned `.ladruno` set draws no spring arrows yet.
