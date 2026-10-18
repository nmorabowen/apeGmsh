### CHANGED — `H5Emitter` methods now live in four mixin modules under `opensees/emitter/`; `h5.py` drops from 5,705 to about 1,800 lines (#2158)

`H5Emitter` keeps its public surface and behaviour. It now inherits four
stateless mixins: `_H5ModelVerbsMixin` (`_h5_verbs_model.py`),
`_H5StagesMixin` (`_h5_stages.py`), `_H5WriteModelMixin`
(`_h5_write_model.py`) and `_H5DeclsMixin` (`_h5_decls.py`), over an
annotation-only host base (`_h5_host.py`). Method bodies are unchanged.
`__all__`, `SCHEMA_VERSION` and `SCHEMA_FLOOR` stay in `h5.py`.
