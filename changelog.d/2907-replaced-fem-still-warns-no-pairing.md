### FIXED — an element read still warns that it has no fem↔ops pairing after `fem=` fails the element check (#2907)

Since #1408, `Results.from_mpco` / `from_ladruno` with `fem=` beside a
`model.h5` that has no element tag map (`fem.to_h5`, or a bridge archive
without `/opensees/element_meta`), and `Results.from_fem`, check the fem's
elements against the capture. On a mismatch they warn
`ModelFemMismatchWarning` and bind the capture's own FEMData. That bound
object was the reader's own snapshot, and `Results` takes the reader's
snapshot to mean that ids need no translation. So every later element read
went silent. Before #1408, each such read warned
`WarnElementTagPairingMissing` ("no fem_eid<->ops-tag pairing"). The
caller's ids are still fem ids, and with no pairing they are read as
OpenSees tags.

Now a mismatch binds a copy of the capture's FEMData, and every element
read warns `WarnElementTagPairingMissing` again. The open still warns
`ModelFemMismatchWarning`, and the bound FEMData still has the capture's
element ids and no physical groups. A matching `fem=`, a bridge archive
with its tag map, and the `model_h5=`-only route are unchanged.
