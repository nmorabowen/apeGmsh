# Design Note: Rigid Element Formulation for node_to_surface

## Problem

`rigidLink` in OpenSees is an **MP_Constraint** (constraint equation), not an element.
When the Transformation handler eliminates slave DOFs, it substitutes
`u_slave = u_master + theta_master x r` into the global equations. But this
never adds stiffness to the master's rotation DOFs. If no element connects
directly to those DOFs, the diagonal of K is zero there — the system is singular.

This manifests as:
- Newton failure under moment loading at the reference node
- Singular K even with no load if rotation DOFs are free and unconnected

The current workaround (`node_to_surface_spring`) replaces `rigidLink` with a
stiff `elasticBeamColumn`. This works because the beam's 12x12 K matrix gives
direct stiffness to both translation and rotation DOFs at both ends. But it is
approximate — the beam has finite (though very large) stiffness.

## Proposed Solution: Rigid Body Element

Formulate the rigid-body kinematics as a **finite element** rather than a
constraint equation. The element encodes:

```
u_slave = u_master + theta_master x r
```

directly in its element stiffness matrix, so K assembly gives the master
rotation DOFs proper diagonal entries.

### Two implementation paths

#### A. Penalty rigid element

An element with K proportional to a penalty parameter alpha:

```
K_e = alpha * G^T G
```

where G is the constraint matrix that maps master DOFs to slave displacements.
For a single master-slave pair with arm vector r = (rx, ry, rz):

```
G = [ I_3  |  -[r x]  |  -I_3  |  0_3x3 ]
```

The element residual is `f = K_e * (u_slave - u_master - theta x r)`.

Pros: simple, no extra DOFs, direct stiffness on all master DOFs.
Cons: penalty sensitivity (too large -> ill-conditioned, too small -> constraint violation).

#### B. Lagrange multiplier rigid element

Add multiplier DOFs lambda that enforce the constraint exactly:

```
[ K_struct   G^T ] [ u      ]   [ f_ext  ]
[ G          0   ] [ lambda  ] = [ 0      ]
```

Pros: exact constraint, no penalty tuning.
Cons: adds DOFs, saddle-point system needs compatible solver, zero diagonal block.

#### C. Augmented Lagrangian

Combine penalty + Lagrange: K_aug = K_struct + alpha * G^T G, plus multiplier update.
Pros: exact in the limit, well-conditioned.
Cons: iterative multiplier update, more complex.

### Recommendation

**Path A (penalty)** is the pragmatic choice for OpenSees:
- OpenSees already uses penalty for other constraint types
- No extra DOFs or solver changes
- Penalty value can be derived from material/geometry (same as stiff beam E*A/L)
- Can be wrapped as a custom element via OpenSeesPy `element('genericClient', ...)`
  or implemented as a Python-side K matrix contribution

The stiff `elasticBeamColumn` in `node_to_surface_spring` is already a penalty
rigid element in disguise — its stiffness is the penalty. A dedicated element
would be cleaner (no need for vec_xz, no geometric length sensitivity) but
mechanically equivalent.

## Scope for Implementation

1. **Custom element class** in apeGmsh that computes K_e from the arm vector r
   and a penalty stiffness derived from the real structure's material properties
2. **Emit via `element()` call** in the OpenSees writer — either as a
   `genericClient` element or by assembling the K contribution externally
3. **Replace `node_to_surface_spring`** stiff beam emission with the rigid element
4. **Fallback**: keep stiff beam as the default since it works and is well-tested;
   offer rigid element as an option

## Relationship to Existing Code

- `node_to_surface` (rigidLink): fails under free-rotation + moment loading
- `node_to_surface_spring` (stiff beam): works, approximate, vec_xz sensitivity
- `node_to_surface_rigid_element` (proposed): works, exact or near-exact, no vec_xz

The phantom node topology remains the same in all three variants:

```
master (ndf=6) --[link]--> phantom (ndf=6) --[equalDOF 1,2,3]--> solid (ndf=3)
```

Only the `[link]` implementation changes.

## When the Issue Does NOT Apply

- **SP-prescribed rotation DOFs**: eliminated from system, no singularity
- **Master connected to frame elements**: frame beam gives rotation stiffness
- **Pure force on translations only**: rotation DOFs are zero but may still be
  singular (no stiffness); works in practice only if the solver tolerates the
  zero pivot (some do, some don't)
