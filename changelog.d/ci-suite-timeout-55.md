### CI: the `suite` job's time limit is 55 minutes, not 40

The slow runner mode noted in #2544 drifted up: most recent green `suite` runs took 34–39 minutes, and #3129 and #3136 were cancelled at 40 minutes with no test failing. 55 minutes leaves room over the slowest green run and still stops a hang long before GitHub's 6-hour limit.
