# dazdiff (Python): roadmap

Status: draft 2026-11-06. Implements [design.md](design.md).
Target: v0.1.0, the first PyPI release.
Tracking: [v0.1.0 (#1)](https://github.com/pedrobtz/py-dazdiff/issues/1), with one sub-issue per
stage; [PyPI-release (#10)](https://github.com/pedrobtz/py-dazdiff/issues/10).

Adapted from the `pkg-roadmap` skill for R packages: GitHub Actions running
pytest replace r-actions, and a PyPI release replaces the CRAN submission.

## Starting point

Uncommitted on `main` as of 2026-11-06: `diff_table()` (benchmark and equal
modes, casting with truncation, measures by type or name, `tolerance` and
`rel_tol`, duplicates `"pair"`/`"error"`, whole-row comparison),
`diff_summary()`, `is_equal()`, the CLI, 13 passing tests, a README and
example CSVs. No CI, no LICENSE file.

Known gaps against [§4](design.md#4-behaviour-of-diff_table), each fixed in
Stage 1 or 2: name normalization doesn't trim or collapse runs; the internal
`KEY_SEQ` column can collide with a user column named `key_seq`; left-over
columns with `by` are silently ignored; no error when no key is in common;
infinities and overflow in `rel_diff` are unhandled; notes and warnings
share one class.

## CI policy

| Profile | Runs | Target |
| --- | --- | --- |
| `quick` | `ruff check`, `ruff format --check`, pytest on Ubuntu, Python 3.13, latest pandas | under 5 minutes |
| `full` | pytest on Ubuntu, macOS and Windows × Python 3.10–3.14; a lowest-direct-dependencies job (`uv sync --resolution lowest-direct`, pandas 2.0); coverage; build sdist and wheel, `twine check`, and run the installed CLI on the example CSVs | — |

Pull requests run `quick`; pushes to `main` and pull requests labelled
`full-ci` run `full`. A stage earns `full` only when the full set checks
something the stage introduces and later stages build on.

## Stages

| Stage | Title | CI | Issue | Pull request | Done |
| --- | --- | --- | --- | --- | --- |
| 0 | [Package identity](#stage-0--package-identity) | quick | [#2](https://github.com/pedrobtz/py-dazdiff/issues/2) | | 2026-11-06 |
| 1 | [Core parity](#stage-1--core-parity) | full | [#3](https://github.com/pedrobtz/py-dazdiff/issues/3) | | |
| 2 | [Duplicates and row keys](#stage-2--duplicates-and-row-keys) | quick | [#4](https://github.com/pedrobtz/py-dazdiff/issues/4) | | |
| 3 | [Text report](#stage-3--text-report) | full | [#5](https://github.com/pedrobtz/py-dazdiff/issues/5) | | |
| 4 | [Test helper](#stage-4--test-helper) | quick | [#6](https://github.com/pedrobtz/py-dazdiff/issues/6) | | |
| 5 | [Full CI](#stage-5--full-ci) | full | [#7](https://github.com/pedrobtz/py-dazdiff/issues/7) | | |
| 6 | [v0.1.0 readiness](#stage-6--v010-readiness) | full | [#8](https://github.com/pedrobtz/py-dazdiff/issues/8) | | |
| 7 | [Review](#stage-7--review) | full | [#9](https://github.com/pedrobtz/py-dazdiff/issues/9) | | |

## Stage 0 — Package identity

**Issue.** [#2](https://github.com/pedrobtz/py-dazdiff/issues/2), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `quick`.

**Goal.** The existing code is on `main`, licensed, and checked by CI on
every pull request.

**Work.**
- Commit the current package, tests, README and examples.
- `LICENSE` (MIT) matching `pyproject.toml`; project URLs and classifiers.
- `.agents/` excluded from the sdist (`[tool.hatch.build.targets.sdist]`).
- `ruff` in the dev group, configured in `pyproject.toml`; the code passes it.
- `.github/workflows/ci.yml` with both profiles from the CI policy, the
  `full-ci` label switch, and a `concurrency` block cancelling superseded
  runs.
- A package `CLAUDE.md`: commands (`uv sync`, `uv run pytest`, `uv run ruff
  check`), the design/roadmap split, and the definition of done.

**Exit criteria.**
- The quick profile is green on the pull request and finishes in under five
  minutes.
- `uv build` produces an sdist without `.agents/`.
- `LICENSE` exists and `pyproject.toml` declares the same license.

**Design.** [§8](design.md#8-dependencies-and-supported-versions),
[§9](design.md#9-packaging-and-release).

## Stage 1 — Core parity

**Issue.** [#3](https://github.com/pedrobtz/py-dazdiff/issues/3), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `full — casting depends on pandas 2 vs 3 string dtypes; the
lowest-dependencies job and Python 3.10 (pandas 2.x) must agree before later
stages build on the result shape`.

**Goal.** `diff_table()` behaves as design §4.1–4.5 and §4.9 say, with the
arguments and result columns that later stages build on.

**Work.**
- Name normalization per §4.2: collapse runs, trim `_`, error on empty
  names, collisions checked only for used `y` columns, errors in the user's
  spelling.
- Lowercase internal columns (`key_seq`), so no user column can collide.
- Left-over columns with `by`: note in benchmark mode, error in equal mode.
- Error when the tables share no key.
- `rel_diff` without overflow; equal infinities are `same`; tolerances apply
  to finite pairs only.
- `DiffNote` for side effects, `DiffWarning` for invented pairing.
- Casting to bool, string and datetime tested on pandas 2 and 3; casting
  errors name the value.
- Keyword-only arguments after `y`; `attrs["shape"]` and `attrs["mode"]`.
- Tests for every mode × `measures` option × `by`, the symmetry of equal
  mode, and that inputs are not modified.

**Exit criteria.**
- Every behaviour listed above has a test, and the full profile is green on
  the pull request, including the lowest-dependencies job.
- `uv run ruff check` is clean.

**Design.** [§4.1–4.5, §4.9](design.md#4-behaviour-of-diff_table).

## Stage 2 — Duplicates and row keys

**Issue.** [#4](https://github.com/pedrobtz/py-dazdiff/issues/4), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `quick`.

**Goal.** Repeated keys and keyless tables are handled as design §4.6–4.8
say, in the library and the CLI.

**Work.**
- Rename `duplicates="pair"` to `"disambiguate"` (settles the
  [§11](design.md#11-open-questions) question; record it there).
- `duplicates="aggregate"`: multiset comparison per key and measure, the
  `groups` shape, and an error when a tolerance is set.
- `row_key="position"` and `"sorted"`, adding a `row_number` key.
- `diff_summary()` handles all three shapes.
- CLI flags `--row-key` and the new `--duplicates` choices.

**Exit criteria.**
- Tests cover each `duplicates` option × shape and both `row_key` modes,
  including a table where every column is a measure.
- The quick profile is green on the pull request.

**Design.** [§4.6–4.8](design.md#46-duplicate-keys).

## Stage 3 — Text report

**Issue.** [#5](https://github.com/pedrobtz/py-dazdiff/issues/5), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `full — Unicode output on Windows consoles is only seen on the
Windows legs, and the CLI's ASCII fallback is built here`.

**Goal.** `diff_report()` and `dazdiff --report` lay out a comparison as
text, bounded in length, in Unicode or ASCII.

**Work.**
- `diff_report()` returning a `DiffReport` with `__str__` and
  `format(unicode=, width=, top_k=, sample_k=)`.
- Sections per §5: header, columns, row alignment diagram, measures table
  with top changes, keys only in `x` / only in `y`.
- CLI `--report` (with `--x-name`/`--y-name`), falling back to ASCII when
  stdout's encoding cannot represent the Unicode output.
- Snapshot tests of the report for the README example, in both character
  sets, plus the three result shapes.

**Exit criteria.**
- Snapshot tests pass, and the full profile is green on the pull request,
  including the Windows legs running `dazdiff --report`.

**Design.** [§5](design.md#5-report), [§7](design.md#7-cli).

## Stage 4 — Test helper

**Issue.** [#6](https://github.com/pedrobtz/py-dazdiff/issues/6), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `quick`.

**Goal.** `assert_table_equal()` lets a test check a table against a
reference with a readable failure.

**Work.**
- `assert_table_equal()` with the defaults in §6; the failure message holds
  `diff_summary()` and the first differing rows, bounded.
- Tests: passing within `rel_tol`, failing on a change, failing on a
  repeated key, and the message's content.
- README sections for the report and the helper.

**Exit criteria.**
- The quick profile is green on the pull request.
- README examples run as written (a test executes them).

**Design.** [§6](design.md#6-test-helper).

## Stage 5 — Full CI

**Issue.** [#7](https://github.com/pedrobtz/py-dazdiff/issues/7), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `full`.

**Goal.** No new features. Everything the full profile finds on `main` is
fixed.

**Work.**
- Start from the full run on the last feature merge, and the comments
  collected on this stage's issue.
- Fix every platform, Python-version and lowest-dependencies failure.
- Coverage floor set in CI at the level reached (at least 90%).

**Exit criteria.**
- A full-profile run is green on the pull request and on the merge commit.

**Design.** [§8](design.md#8-dependencies-and-supported-versions).

## Stage 6 — v0.1.0 readiness

**Issue.** [#8](https://github.com/pedrobtz/py-dazdiff/issues/8), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `full`.

**Goal.** The package is ready to publish.

**Work.**
- Settle the distribution name ([§11](design.md#11-open-questions)).
- `version = "0.1.0"` in `pyproject.toml` and `__version__` from package
  metadata (one source); `CHANGELOG.md` with a 0.1.0 section.
- `.github/workflows/release.yml`: build, then publish with PyPI trusted
  publishing on a `v*` tag; a manual TestPyPI dispatch.
- A TestPyPI upload, installed into a clean environment, running the CLI on
  the examples.
- README: installation from PyPI, badges.

**Exit criteria.**
- `twine check` passes on the built sdist and wheel.
- The TestPyPI package installs and `dazdiff --version` prints 0.1.0.
- A full-profile run is green on the merge commit.

**Design.** [§9](design.md#9-packaging-and-release).

## Stage 7 — Review

**Issue.** [#9](https://github.com/pedrobtz/py-dazdiff/issues/9), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazdiff/issues/1).

**CI.** `full`.

**Goal.** No new features. The release candidate is reviewed twice.

**Work.**
1. Critical review of code, tests and style (the `/code-review` and
   `review-testing` skills). Blocking and required findings become
   sub-issues of this issue, each fixed in a pull request that closes it.
2. Packaging review after those fixes: metadata, sdist and wheel contents,
   README rendering on PyPI, supported versions matching CI.

**Exit criteria.**
- Every blocking and required finding closed; packaging review clean;
  `CHANGELOG.md` reflects user-visible fixes.
- A full-profile run green on `main`, whose head becomes the release commit.

No pull request closes this issue; it closes when both reviews are done.

## After 0.1.0

From the daffiz roadmap, in its priority order:

- Row accessors (`original_rows()`, `x_only()`, `y_only()`), taking the
  inputs back: H in daffiz.
- Report in `assert_table_equal()`'s failure text: H.
- Tolerance per measure (a mapping for `tolerance` / `rel_tol`): M.
- `exclude=` columns: M.
- Lossless int/float widening in equal mode for the library (the CLI already
  widens CSV columns, [§7](design.md#7-cli)): M.
- `diff_changes()`, change counts per measure as data: M.
- Non-numeric measures (text, bool, dates), opt-in: M.
- Projected-size warning before large comparisons: M.
- Parquet input for the CLI through an optional `pyarrow` extra: L.
- JSON export of the report, hidden columns, slim report: L.

## Blocking open questions

- [§11](design.md#11-open-questions) (distribution name) blocks Stage 6.
