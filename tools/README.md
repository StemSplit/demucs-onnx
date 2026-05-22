# `tools/` — internal release tooling

Files in this directory are **not** shipped to PyPI. They are scoped to
the monorepo + the public `StemSplit/demucs-onnx` repo as internal
release infrastructure. The `[tool.hatch.build.targets.sdist]` include
list in `pyproject.toml` covers `src/`, `assets/`, `examples/`, `docs/`,
`mkdocs.yml`, `README.md`, `CHANGELOG.md`, `LICENSE`, and
`pyproject.toml` — `tools/` is deliberately excluded.

## `publish.py`

Safe publish flow for the public `StemSplit/demucs-onnx` GitHub repo.
Root-cause fix for the v0.3.3 monorepo-leak incident (2026-05-22).

Hard guardrails it enforces, in order:

1. **Subtree-extract → temp dir** — copies `scripts/demucs-onnx-pkg/`
   (this package's directory) into a fresh `mktemp -d`. Never pushes
   from the monorepo working tree.
2. **Forbidden-file scan** — refuses to push if `.env*`, `*.pem`,
   `*.key`, or `.tmp_*.py` ended up in the staged tree.
3. **No AI/agent attribution** — builds the commit via `git commit-tree`
   plumbing (bypasses any shell wrapper that auto-injects
   `Co-authored-by` trailers) and verifies the resulting message
   contains no `Co-authored-by:` / `Generated-by:` / `Signed-off-by:`
   trailers.
4. **File-count guardrail** — runs
   `git diff-tree --no-commit-id --name-only -r --root HEAD | wc -l`
   and aborts if > `DEMUCS_ONNX_PUBLISH_MAX_FILES` (default 100, hard
   floor 1 — cannot be disabled). The v0.3.3 push would have tripped
   at 2,638 files.
5. **Remote-URL allowlist** — refuses to push unless `origin` matches
   `https://github.com/StemSplit/demucs-onnx(.git)?` or
   `git@github.com:StemSplit/demucs-onnx(.git)?`.
6. **Default `--dry-run`** — printing the full plan but never pushing.
   Requires `--no-dry-run` or `PUBLISH_LIVE=1` to actually push or call
   `gh release create`.

### Usage

```bash
# Plan only (no push). Default.
python scripts/demucs-onnx-pkg/tools/publish.py

# Actually publish.
PUBLISH_LIVE=1 python scripts/demucs-onnx-pkg/tools/publish.py
# or
python scripts/demucs-onnx-pkg/tools/publish.py --no-dry-run

# Tighten the file-count cap for a more paranoid release:
DEMUCS_ONNX_PUBLISH_MAX_FILES=80 \
  python scripts/demucs-onnx-pkg/tools/publish.py --no-dry-run
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | success (dry-run or live) |
| `2` | a guardrail tripped (file count, remote URL, forbidden file, version mismatch, hard floor, …) |
| `3` | subprocess failure during `git`/`gh` invocation |

### Author identity

Every commit and tag the script produces is authored as
`StemSplit <team@stemsplit.io>`. There is no fallback to
`git config user.{name,email}`, so the script behaves identically on
any developer's machine, in CI, and inside an agent shell wrapper.

### What the script does NOT do

- It does NOT publish to PyPI. `uv publish` / `hatch publish` continues
  to be invoked manually with `UV_PUBLISH_TOKEN` from
  `scripts/demucs-onnx-pkg/.env.local`. The leak vector being closed
  here is the GitHub side; PyPI was verified clean during the incident
  (its sdist includes are scoped, so the monorepo working tree never
  propagated into a release artifact).
- It does NOT re-deploy the docs site. The docs site auto-deploys from
  the `.github/workflows/docs.yml` workflow on push to `main` of the
  public repo.
- It does NOT manage the v0.1.0–v0.3.3 historical tags. Those were
  re-anchored during the incident remediation; this script only
  publishes the next release.

## `test_publish_safety.sh`

Verification suite for `publish.py`. Always runs in dry-run mode; never
touches a remote.

```bash
bash scripts/demucs-onnx-pkg/tools/test_publish_safety.sh
```

Five cases:

| # | Scenario | Expected exit | Expected output |
|---|---|---|---|
| 1 | Real `demucs-onnx-pkg`, default cap | `0` | `file count: …` |
| 2 | Fake pkg with 105 files | `2` | `FILE-COUNT GUARDRAIL TRIPPED` |
| 3 | Real pkg, attacker remote URL | `2` | `REMOTE-URL GUARDRAIL TRIPPED` |
| 4 | `DEMUCS_ONNX_PUBLISH_MAX_FILES=0` | `2` | `below the hard floor` |
| 5 | Real pkg, `DEMUCS_ONNX_PUBLISH_MAX_FILES=200` | `0` | `max_files: 200` |

Wire this into CI on the monorepo side to keep the guardrails
regression-tested.
