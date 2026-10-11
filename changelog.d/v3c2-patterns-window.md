### ADDED — apeGmshViewer: the Patterns & series window shows every load pattern and plots every time series (V3c2, #1643)

The **Patterns & series** window now lists every load pattern under the
stage that adds it (or `global`), titled by its name, or by its `#k` key
when it has none. Each pattern shows its type and tag, its time series as a
link (the `series` field's reference from `/opensees/decl_params`), and its
other parameters by field name. It also gives each archive table with its
row count (nodal loads, single-point constraints, element loads, held DOFs)
and the targets the archive names, such as `pg Col`. Every time series lists
its parameters, the patterns it drives, and a small line plot: Constant and
Linear show their factor, and Path and Trig are drawn from their fields. Any
other type gets a sentence instead of a guessed curve. When a series or
pattern has a recorded source, a click on it opens that source. A file
without `/opensees/decls` or `decl_params` says so, and the window never
rebuilds parameters from positional arguments.
