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
uv run --python 3.13 --with great-docs==0.17.0 great-docs build    # site into great-docs/_site
uv run --python 3.13 --with great-docs==0.17.0 great-docs preview  # serve it locally
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
- The documentation site is `great-docs.yml` (API reference groups),
  `user_guide/*.qmd` (guide pages) and the docstrings; a new public name
  goes into a `reference` group. Guide output is pasted from real runs.
- Every user-visible change gets a line under the top section of
  `CHANGELOG.md`, `## Unreleased`.

## Definition of done

Tests and `ruff check` / `ruff format --check` pass locally, the pull
request's checks are green at the stage's tier, and the roadmap's stage
table and the design are updated in the same pull request.

Merge pull requests with a merge commit (`gh pr merge --merge`), and only
when told to or when running the roadmap loop the user approved.

## Releasing

Merges to `main` publish nothing. Pushing a `v*` tag runs
`.github/workflows/release.yml`, whose `pypi` job waits for the user to
approve it in the `pypi` environment. Only then does it upload to PyPI.

When the user says "let's release" (optionally naming a version), do the
steps below. That phrase approves merging the release pull request and
pushing the tag, nothing more.

1. **Check.** `main` is up to date with `origin/main` and clean; the latest
   `full` CI run on `main` is green. If not, stop and report.
2. **Version.** The version is `project.version` in `pyproject.toml`
   without its `.devN` suffix (`0.1.1.dev0` → `0.1.1`), unless the user
   named one. Say which version you are releasing.
3. **Release pull request** `Release X.Y.Z`, from branch `release-X.Y.Z`:
   set `project.version = "X.Y.Z"`; rename the top `CHANGELOG.md` heading
   to `## X.Y.Z (YYYY-MM-DD)` with today's date. Label it `full-ci`, wait
   for green, merge with `gh pr merge --merge`.
4. **Tag** the merge commit on `main`, then push it:
   `git tag -a vX.Y.Z -m "dazdiff X.Y.Z"` and `git push origin vX.Y.Z`.
5. **Approval.** Give the user the release run's URL and say it waits for
   their approval (Review deployments → `pypi` → Approve). Never approve it
   yourself. Watch the run to the end.
6. **Verify.** In a clean venv in the scratchpad,
   `pip install dazdiff==X.Y.Z` from PyPI; `dazdiff --version` prints
   X.Y.Z; the CLI on `examples/` exits 1.
7. **GitHub release.** `gh release create vX.Y.Z --title "dazdiff X.Y.Z"`
   with that version's `CHANGELOG.md` section as notes; check it with
   `gh release view vX.Y.Z`.
8. **Next version.** Pull request `Start X.Y.(Z+1).dev0`: set that version and
   add an empty `## Unreleased` heading at the top of `CHANGELOG.md`; merge
   on green. Close the version's issue, if there is one.

Never move, delete or re-push a tag, and never re-upload a version: PyPI
refuses a version twice. If the run fails before PyPI received the files,
stop and ask; if it fails after, fix forward with the next patch version.
To rehearse on TestPyPI (which deploys only from `main`, and already has
0.1.0), merge a pre-release version (`X.Y.Zrc1`) to `main`, run
`release.yml` by hand, and set the version back before step 3.
