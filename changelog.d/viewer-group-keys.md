### FIXED — apeGmshViewer: reading a model is no longer quadratic in a group's children (#2835)

The model reader lists each open HDF5 group once instead of once per child. A
child that is listed but cannot be opened is still refused by name.
