### CHANGED — apeSees procedure methods now live in mixins under opensees/procedures/ (program slice S1-b, #1450)

A pure move: `eigen`, the modal, frequency-response, explicit and contact-query procedures moved, bodies
unchanged, into underscore mixin modules that `apeSees` inherits. The public surface is unchanged.
