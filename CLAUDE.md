# dazdiff (Python)

A Python port of the R package dazdiff (`pedrobtz/dazdiff`, formerly daffiz): compare two
tables value by value, as a library on pandas DataFrames and as a CLI on CSV
files.

## Where things are decided

- `.agents/design.md` is the single source of truth for what the package is
  and why: API, behaviour, divergences from the R package, open questions. A
  decision changes the design, with a dated one-line reason.
- `.agents/roadmap.md` owns the order of work: stages, exit criteria, CI
  tiers, issue links. Each stage is a sub-issue of `v0.1.0` (#1), closed by
  one pull request titled like the stage, with `Closes #<n>`.

## Commands

```sh
uv sync                       # install, including the dev group
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv build                      # sdist and wheel into dist/
uv run dazdiff examples/benchmark.csv examples/stored.csv --by id,region
```

## CI

`.github/workflows/ci.yml` has two profiles: `quick` on pull requests (lint,
pytest on Ubuntu / 3.13) and `full` on pushes to `main`, manual runs, and
pull requests labelled `full-ci`. The roadmap says which tier each stage
merges on.

## Rules

- Source lives in `src/dazdiff/`; `core.py` is the library, `report.py` the
  text report, `cli.py` the command. Public names are re-exported from `__init__.py`.
- Behaviour follows the R package unless `design.md` §2.1 records a divergence.
- The caller's DataFrames are never modified.
- Every user-visible change gets a line under the top section of
  `CHANGELOG.md` (`## 0.1.0 (unreleased)` until 0.1.0 ships).

## Definition of done

Tests and `ruff check` / `ruff format --check` pass locally, the pull
request's checks are green at the stage's tier, and the roadmap's stage
table and the design are updated in the same pull request.

Merge pull requests with a merge commit (`gh pr merge --merge`), and only
when told to or when running the roadmap loop the user approved.
