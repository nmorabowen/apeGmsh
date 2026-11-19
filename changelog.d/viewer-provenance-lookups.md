### FIXED — apeGmshViewer: provenance lookups at load are no longer quadratic (#2837)

Opening a model searched the `/provenance` records once per record, twice over.
The load loops now use the row they are on, and go-to-source finds a listing
entry by key.
