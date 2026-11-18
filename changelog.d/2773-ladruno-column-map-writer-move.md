### CHANGED — the `.ladruno` column-map writer moves into `_ladruno_writer` (#2773, ADR 0131 D10 stage 0)

`_write_column_map`, `_meta_column`, `_split_block` and `MPCOConversionRefused` move from
`results/readers/_ladruno_from_mpco.py` to `results/readers/_ladruno_writer.py`, so the
converter imports its writer from one module. No behaviour changes: the public import
`apeGmsh.results.MPCOConversionRefused` is the same class, now defined in the writer module.
