### Removed dead station-order check (#2337)

`require_ordered_stations` in `results/readers/_out_station_slicing.py` is deleted. Since #2331 and #2335 a model-less `.out` line-station read is refused by name before any slice, so no route reached it. No behaviour changes.
