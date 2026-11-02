# Contributing to DL+DiReCT

This document describes the branch model and version lifecycle used in this repo:
how `dev` and `main` relate, how contributors get changes in, and how versions are
bumped and released.

# Branch model

- **`main`** is the stable branch and the repository's default branch, so a plain
  `git clone` (as in the [README](README.md#install-dldirect)) installs the latest
  release. It only moves when a finished `dev` cycle is promoted into it.
- **`dev`** is the integration branch. All day-to-day work happens on branches
  cut from `dev`, and lands back on `dev` via pull request.
- Contributors don't branch from or push directly to `main`.
- This is enforced, not just documented: a required CI check
  ([`.github/workflows/guard-main-base.yml`](.github/workflows/guard-main-base.yml))
  fails any pull request targeting `main` whose head branch isn't `dev`.
- There is no automated test of the pipeline itself (it needs a GPU, FreeSurfer
  and real scans): run it on real data before opening a PR that changes
  processing.

To work on the current development version:
```bash
git clone https://github.com/SCAN-NRAD/DL-DiReCT-V2.git
cd DL-DiReCT-V2
git checkout dev
git checkout -b <your-branch>
pip install -e .
```
Use `python -m pip` if `pip` might not belong to the active environment, and check
with `python -m pip show DL-DiReCT` that the install points at your checkout.

# Versioning

The package version is the `version` field in [`pyproject.toml`](pyproject.toml).

- On `dev`, it carries a [PEP 440](https://peps.python.org/pep-0440/) dev-release
  suffix, e.g. `1.5.0.dev0`.
- On `main`, it is a clean release version, e.g. `1.5.0`.

This is enforced, not just documented: a required CI check
([`.github/workflows/version-check.yml`](.github/workflows/version-check.yml)) parses
the version as PEP 440 and fails a PR targeting `main` if it still carries a dev
suffix, or a PR targeting `dev` if it doesn't. The latter also catches a branch
accidentally cut from `main` instead of `dev` (it would inherit `main`'s clean
version). On a PR into `main` it also fails if the release tag for that version
already exists.

# The development cycle

1. **Branch `dev` from `main`.** Bump `version` in `pyproject.toml` (minor/micro as
   appropriate for what the cycle is expected to deliver) with a dev suffix,
   e.g. `1.5.0.dev0`.
2. **Contributors branch from `dev`** for each change (feature, fix, refactor),
   push that branch, and open a pull request **into `dev`**. Keep unrelated work
   on separate branches and PRs.
3. PRs get reviewed, amended, and eventually merged into `dev`. As work lands,
   bump the dev suffix whenever it's useful to tell one in-progress state from
   another (e.g. `.dev0` → `.dev1`), for instance before processing a dataset
   with the current `dev`, so its results can be traced back to that exact state.
4. **When `dev` is ready to ship:** bump the version one last time without the
   dev suffix (e.g. `1.5.0.dev1` → `1.5.0`) in a pull request into `dev` that
   changes nothing else. `version-check` accepts a clean version on a PR into
   `dev` only in this case: the release of `dev`'s current `.devN` version. Then
   open a pull request from `dev` into `main`, and merge it with a merge commit.
5. **The merge commit on `main` gets tagged automatically** with the release
   version (e.g. `v1.5.0`) by
   [`.github/workflows/tag-release.yml`](.github/workflows/tag-release.yml), which
   also publishes the GitHub release page for that tag, with notes listing the
   PRs merged since the previous release. No manual tagging needed; edit the
   release notes on GitHub if they need more context.
6. **Delete the merged `dev` branch**, then branch a new `dev` from the now-updated
   `main` and bump its version (step 1) to start the next cycle.

```
main ----A-------------------------------F--------------------->
          \                             / (PR: dev -> main, tag v1.5.0)
           \                           /
dev         B --- C --- D --- E ------      (deleted after merge)
            (1.5.0.dev0)  (…dev1)  (1.5.0, pre-merge bump)

new dev ------------------------------------- G --->  (branched from F, 1.6.0.dev0)
```
