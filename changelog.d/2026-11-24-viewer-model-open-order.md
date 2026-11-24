### FIXED — apeGmshViewer keeps the model on screen as the open set when an older open finishes last

Dropping a large model and then a small one could leave the viewer pairing
results against the large model after the small one was on screen: the large
model's read finished last and main made it the open set again, so the results,
geometry and particle doors refused the small model's files. Main now moves the
open set only for the latest `model:open`, as the page already keeps only the
latest answer.
