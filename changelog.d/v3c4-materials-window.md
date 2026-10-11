### ADDED — apeGmshViewer: the Materials & sections window (V3c4, #1645)

The **Materials & sections** window replaces its V3c0 placeholder. It lists
every uniaxial and nD material, every section and every beam integration by
its name (or its `#k` key when unnamed), with its OpenSees type, its tag as
evidence, and its parameters by field name from `/opensees/decl_params`.
Each `{"$decl": key}` parameter (a section's patch materials, an
integration's sections, the three sections of a Hinge integration) is a link
that selects that declaration. Each row also says what uses it: the elements
and objects whose definition-chain links point at it, and the declarations
whose `decl_params` name it. A file written before opensees 2.26.0 says that
parameters by field name are not in it; the window does not read the
positional arguments as names.
