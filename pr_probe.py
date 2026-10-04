import h5py, numpy as np
with h5py.File("tests/fixtures/schema_corpus/opensees_2.20.h5","r") as f:
    for t in f["opensees/element_meta"]:
        g=f["opensees/element_meta"][t]; print(t, list(g), g["args"][()].tolist(), g["fem_eids"][()].tolist())
    for t in f["opensees/transforms"]:
        g=f["opensees/transforms"][t]; print(t, dict(g.attrs), g["per_element_vecxz"][()].tolist(), g["per_element_emitted_tag"][()].tolist())
    print(f["nodes/coords"][()].tolist())
