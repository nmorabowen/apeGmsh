### FIXED — apeGmshViewer follows the model you opened last when two opens overlap

If you dropped a large model and then a small one, the large one could finish
last. Main then paired results, geometry and particle reads against the model you
had left, though the page showed the newer model. Main now keeps the set of the
latest open. A material-tester run that a failed or closed model cancelled can no
longer finish a later run with its late answer: run ids keep counting across
those switches, and answers are matched on their sequence number.
