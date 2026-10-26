### Fixed

- `LadrunoReader.read_line_stations`: pass the raw recorded element tags (`element_index`) rather than the translated `fem_index` to `_line_station_quaternions` so `LineStationSlab.local_axes_quaternion` resolves `MODEL/LOCAL_AXES` rows when `model_h5` attaches an `ElementTagTranslator` with `fem_id != ops_tag`, translate `LadrunoReader.read_local_axes` return keys to `fem_eid` under `model_h5`, and declare `MODEL/LOCAL_AXES` tags in `_recorded_element_tags`.
