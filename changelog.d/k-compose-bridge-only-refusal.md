### FIXED — composing a results file from a bridge-only archive raises `BridgeOnlyArchiveError` (#1881)

`NativeWriter.open(fem=..., model_h5_src=...)` copies the source's
`/opensees` zone and forwards its `ndm`, which it resolves from the source's
node coordinates. A bridge-only source, one with `/meta` and `/opensees` but
no neutral zone (for example an archive written from a `FEMStub`), has no
`nodes` group. The compose used to fail with h5py's raw `KeyError` after the
`/opensees` copy had already landed in the destination. It now raises
`BridgeOnlyArchiveError` (the class `OpenSeesModel.from_h5` raises for the
same archive since #1859), names the zones present, and refuses before
anything is copied. A source with a neutral zone composes as before, and so
does a bridge-only source opened without `fem=` (nothing reads its nodes).
