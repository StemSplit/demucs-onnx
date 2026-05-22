# Publishing `demucs-onnx`

| | |
|---|---|
| PyPI package | [`demucs-onnx`](https://pypi.org/project/demucs-onnx/) |
| TestPyPI | [`test.pypi.org/project/demucs-onnx`](https://test.pypi.org/project/demucs-onnx/) |
| GitHub repo | <https://github.com/StemSplit/demucs-onnx> |
| Release workflow | [`.github/workflows/publish.yml`](.github/workflows/publish.yml) |
| Manual fallback | [`tools/publish.py`](tools/publish.py) (break-glass only) |

This file documents the **canonical release flow**. The workflow file is
the source of truth; this README is the operator-facing index.

## TL;DR — cut a release

```bash
# 1. Bump versions in three places (they MUST agree — verify gate enforces it).
#    - src/demucs_onnx/__init__.py        →  __version__ = "X.Y.Z"
#    - pyproject.toml                     →  version = "X.Y.Z"
#    - CHANGELOG.md                       →  prepend a new `## [X.Y.Z] - YYYY-MM-DD` section

# 2. Commit + tag + push the tag. CI does the rest.
git add src/demucs_onnx/__init__.py pyproject.toml CHANGELOG.md
git commit -m "chore: release X.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

The tag push triggers `publish.yml`, which: lints across Python 3.10/3.11/3.12/3.13,
runs a 1-second parity smoke on CPU, builds the sdist + wheel, publishes to
PyPI via OIDC (no API tokens), attests build provenance with sigstore, and
opens a GitHub Release with auto-generated notes and the dist files attached.

## Dry-run flow (TestPyPI)

Validate a release end-to-end before tagging — recommended whenever you've
changed `pyproject.toml`, `__version__`, the build, or the parity gate:

```bash
gh workflow run publish.yml -f dry_run=true
```

This skips the real PyPI push and instead publishes to
[TestPyPI](https://test.pypi.org/project/demucs-onnx/) using the
`testpypi` GitHub environment. `dry_run` defaults to `true`, so a bare
`gh workflow run publish.yml` is also a dry-run.

To install from TestPyPI for a final smoke:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  demucs-onnx==X.Y.Z
demucs-onnx --version
```

## Pre-tag checklist

1. **Bump `__version__`** in `src/demucs_onnx/__init__.py`.
2. **Bump `version`** in `pyproject.toml`. Must match step 1 exactly — the
   verify gate hard-fails otherwise.
3. **Prepend a `CHANGELOG.md` entry** under
   `## [X.Y.Z] - YYYY-MM-DD — <short subject>` following the existing
   Keep-a-Changelog format.
4. **Run the dry-run** (`gh workflow run publish.yml -f dry_run=true`) and
   confirm it goes green.
5. **Commit + tag + push.** Tag format is `vX.Y.Z` (the leading `v` is
   required — the workflow trigger filters on `v*.*.*`).

The verify gate cross-checks `__version__` against `pyproject.toml` and
against the pushed tag (with the leading `v` stripped) on every job.

## One-time setup

Already configured on the StemSplit org; documented here so the workflow is
reproducible if the org is rebuilt.

### PyPI Trusted Publishing

At <https://pypi.org/manage/account/publishing/>:

| Field | Value |
|---|---|
| PyPI Project Name | `demucs-onnx` |
| Owner | `StemSplit` |
| Repository name | `demucs-onnx` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Repeat at <https://test.pypi.org/manage/account/publishing/> with environment
name `testpypi` for the dry-run path.

PyPI's docs: <https://docs.pypi.org/trusted-publishers/>.

### GitHub Environments

Repo Settings → Environments → "New environment":

| Environment | Recommended protection rules |
|---|---|
| `pypi` | Require manual approval from at least one maintainer. Restrict to tag refs `v*.*.*`. |
| `testpypi` | No approval required. Restrict to the `publish.yml` workflow. |

### Secrets

**None required.** Trusted Publishing handles auth via OIDC short-lived
tokens; the workflow file does not need (and must not be granted) a
long-lived `PYPI_API_TOKEN`.

## Emergency manual fallback — `tools/publish.py`

`tools/publish.py` is the **break-glass** release path. It exists for one
reason: if GitHub Actions is broken or compromised and a release has to ship
anyway, a maintainer can run it locally with strict guardrails.

What it does:

- Subtree-extracts the package into a fresh `mktemp -d` so the monorepo
  working tree is never the source of the push.
- Scans for forbidden files (`.env*`, `*.pem`, `*.key`, `.tmp_*.py`).
- Enforces the file-count cap (default 100, hard floor 1).
- Allow-lists the `origin` URL against
  `https://github.com/StemSplit/demucs-onnx`.
- Builds the commit via `git commit-tree` plumbing (so no shell wrapper
  can inject `Co-authored-by:` / `Generated-by:` trailers).
- Defaults to `--dry-run`. Requires `--no-dry-run` (or `PUBLISH_LIVE=1`) to
  actually push.

What it does **not** do:

- It does NOT publish to PyPI. PyPI is `uv publish` from the maintainer's
  machine with `UV_PUBLISH_TOKEN` from `.env.local`. **Do not commit that
  token.**
- It does NOT redeploy the docs site. `docs.yml` handles that on push to
  `main`.

See [`tools/README.md`](tools/README.md) for the full exit-code table and
the regression test in `tools/test_publish_safety.sh`.

> **Use the CI path whenever possible.** The manual path exists for
> incident response; treat it as such.

## Files in this directory

- This directory is a **standalone git repo** wired to
  `https://github.com/StemSplit/demucs-onnx.git`. It happens to live inside
  the musicai monorepo for convenience — commits made here push to the
  public repo, not to the monorepo.
- See [`CLAUDE.md`](../../CLAUDE.md) in the monorepo for the multi-repo
  ground rules.
