# modeltrees

Model-based recursive partitioning (MOB) in Python, ported from R's
partykit.

## Development

```
uv sync                 # the package and the dev group
uv run pytest           # default run, slow tests excluded
uv run pytest -m ""     # every test
uv run pytest -m slow   # only the slow tests
```

Tests marked `slow` take more than about 5 seconds each or as a
parametrized group: growing the larger negative binomial and multinomial
fixture trees, the parallel fixture trees and the benchmark tree. They are
excluded by default through `addopts` in pyproject.toml.
