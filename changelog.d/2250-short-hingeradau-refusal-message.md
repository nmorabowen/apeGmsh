### CHANGED — a short-HingeRadau fibre read names the repeated or out-of-order station (#2250)

The `.out` fibre read still refuses an element whose `integrationPoints`
stations are repeated or out of order, as a HingeRadau rule with
`lp_i + lp_j >= L/4` writes them. The message used to blame the record split;
it now names the element, its stations and the HingeRadau cause.
