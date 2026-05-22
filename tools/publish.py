"""Safe publish flow for the public ``demucs-onnx`` GitHub repo.

Root-cause fix for the v0.3.3 monorepo-leak incident (2026-05-22). The
prior publish path was ad-hoc — the worker ran ``git push -f`` from
inside the StemSplit/Musicai monorepo working tree, which on v0.3.3
ended up force-pushing 2,638 monorepo files (including ``apps/``,
``packages/``, internal ``.cursor/`` rules, deploy scripts with
plaintext production credentials) to the public ``StemSplit/demucs-onnx``
repository on GitHub.

This script codifies the safe pattern that v0.1.0 → v0.3.2 used by
accident, with hard guardrails that make a repeat of the v0.3.3
mistake impossible:

  1. Always extracts the ``scripts/demucs-onnx-pkg/`` subtree to a
     fresh temporary directory and pushes from there. NEVER pushes
     from the monorepo working tree itself.
  2. File-count guardrail: refuses to push if the staged tree has more
     than ``DEMUCS_ONNX_PUBLISH_MAX_FILES`` files (default 100, hard
     floor of 1 — cannot be disabled entirely).
  3. Remote-URL safety check: refuses to push unless ``origin`` resolves
     to ``StemSplit/demucs-onnx`` exactly.
  4. Default is ``--dry-run``. ``--no-dry-run`` (or ``PUBLISH_LIVE=1``)
     is required to actually push or call ``gh release create``.
  5. Author is hardcoded ``StemSplit <team@stemsplit.io>``; commits are
     built via ``git commit-tree`` plumbing to bypass any shell wrapper
     that auto-injects ``Co-authored-by`` trailers.

Usage::

    # Dry run (default): print plan, do not push, exit 0 on success.
    python scripts/demucs-onnx-pkg/tools/publish.py

    # Live publish:
    PUBLISH_LIVE=1 python scripts/demucs-onnx-pkg/tools/publish.py
    # or
    python scripts/demucs-onnx-pkg/tools/publish.py --no-dry-run

    # Override file-count cap (must stay >= 1):
    DEMUCS_ONNX_PUBLISH_MAX_FILES=80 python .../publish.py --no-dry-run

Exit codes:
  0  success (dry-run or live)
  2  guardrail tripped (file count, remote URL, version mismatch, etc.)
  3  subprocess failure during git/gh invocation
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PKG_DIR_DEFAULT = Path(__file__).resolve().parent.parent

ALLOWED_REMOTE_PATTERNS = [
    re.compile(r"^https://github\.com/StemSplit/demucs-onnx(\.git)?/?$"),
    re.compile(r"^git@github\.com:StemSplit/demucs-onnx(\.git)?$"),
]

AUTHOR_NAME = "StemSplit"
AUTHOR_EMAIL = "team@stemsplit.io"

GITIGNORE_LIKE_EXCLUDES = [
    ".git/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    "*.egg-info/",
    ".pytest_cache/",
    ".ruff_cache/",
    "build/",
    "dist/",
    ".coverage",
    ".env",
    ".env.local",
    ".env.*.local",
    "out/",
    "artifacts/",
    "*.onnx",
    "*.wav",
    ".tmp_*.py",
    "site/",
    "_site_test/",
    "scripts/",
    "node_modules/",
    ".DS_Store",
    "*.pem",
    "*.key",
    "secrets/",
]

DEFAULT_MAX_FILES = 100
HARD_FLOOR_MAX_FILES = 1


class PublishError(Exception):
    pass


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = True,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with sane defaults. Always text mode."""
    print(f"  $ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        capture_output=capture,
        text=True,
        input=stdin.decode() if stdin else None,
    )
    if check and result.returncode != 0:
        print(f"    -> exit {result.returncode}", file=sys.stderr)
        if result.stdout:
            print(f"    stdout: {result.stdout}", file=sys.stderr)
        if result.stderr:
            print(f"    stderr: {result.stderr}", file=sys.stderr)
        raise PublishError(f"command failed: {' '.join(cmd)}")
    return result


def read_version_from_pyproject(pkg_dir: Path) -> str:
    pyproject = pkg_dir / "pyproject.toml"
    if not pyproject.exists():
        raise PublishError(f"missing {pyproject}")
    for line in pyproject.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"\s*$', line)
        if m:
            return m.group(1)
    raise PublishError(f"could not find version in {pyproject}")


def extract_changelog_for(pkg_dir: Path, version: str) -> tuple[str, str]:
    """Return ``(heading_subtitle, body)`` for the CHANGELOG section
    matching ``version``. The heading subtitle is the text after the date
    in lines like ``## [0.3.3] - 2026-05-21 — Cross-link the official Python SDK``
    (subtitle = ``Cross-link the official Python SDK``)."""
    changelog = pkg_dir / "CHANGELOG.md"
    if not changelog.exists():
        return "", ""
    text = changelog.read_text()
    pattern = re.compile(
        rf"^(## \[?{re.escape(version)}\]?[^\n]*)\n(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return "", ""
    heading = m.group(1)
    body = m.group(2).strip()
    subtitle_match = re.search(r"—\s+(.+?)\s*$", heading)
    subtitle = subtitle_match.group(1).strip() if subtitle_match else ""
    return subtitle, body


def copy_subtree(src: Path, dst: Path) -> None:
    """Copy ``src/`` into ``dst/`` while honoring GITIGNORE_LIKE_EXCLUDES.

    Implemented with shutil.copytree + ignore callback so it has no
    dependency on rsync (which is BSD-licensed on macOS but absent on
    minimal CI images).
    """
    if dst.exists():
        shutil.rmtree(dst)

    glob_excludes = [p for p in GITIGNORE_LIKE_EXCLUDES if not p.endswith("/")]
    dir_excludes = {p.rstrip("/") for p in GITIGNORE_LIKE_EXCLUDES if p.endswith("/")}

    def ignore(dir_path: str, names: list[str]) -> list[str]:
        ignored: list[str] = []
        for name in names:
            if name in dir_excludes:
                ignored.append(name)
                continue
            for pat in glob_excludes:
                if Path(name).match(pat):
                    ignored.append(name)
                    break
        return ignored

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=False)


def init_orphan_repo(repo_dir: Path) -> None:
    _run(["git", "init", "-b", "main", "-q"], cwd=repo_dir)
    _run(["git", "add", "."], cwd=repo_dir)


def build_commit(
    repo_dir: Path,
    *,
    subject: str,
    body: str,
    author_date_iso: str,
    parent: str | None = None,
) -> str:
    """Build a commit via ``git commit-tree`` plumbing (bypasses any
    shell-wrapper-injected trailers like Co-authored-by)."""
    tree = _run(["git", "write-tree"], cwd=repo_dir).stdout.strip()
    msg_path = repo_dir / ".git" / "COMMIT_TREE_MSG"
    msg = subject + ("\n\n" + body if body else "") + "\n"
    msg_path.write_text(msg)

    cmd = ["git", "commit-tree", "-F", str(msg_path), tree]
    if parent:
        cmd[2:2] = ["-p", parent]

    env = {
        "GIT_AUTHOR_NAME": AUTHOR_NAME,
        "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
        "GIT_AUTHOR_DATE": author_date_iso,
        "GIT_COMMITTER_NAME": AUTHOR_NAME,
        "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
        "GIT_COMMITTER_DATE": author_date_iso,
    }
    sha = _run(cmd, cwd=repo_dir, env=env).stdout.strip()
    _run(["git", "update-ref", "refs/heads/main", sha], cwd=repo_dir)
    return sha


def build_annotated_tag(
    repo_dir: Path,
    *,
    name: str,
    annotation_subject: str,
    annotation_body: str,
    tagger_date_iso: str,
    target: str,
) -> None:
    """Create an annotated tag with a verbatim message (no autotrailer)."""
    msg_path = repo_dir / ".git" / "TAG_MSG"
    msg = annotation_subject + ("\n\n" + annotation_body if annotation_body else "") + "\n"
    msg_path.write_text(msg)

    env = {
        "GIT_COMMITTER_NAME": AUTHOR_NAME,
        "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
        "GIT_COMMITTER_DATE": tagger_date_iso,
    }
    _run(
        [
            "git", "tag", "-a", "-F", str(msg_path),
            "--cleanup=verbatim", name, target,
        ],
        cwd=repo_dir,
        env=env,
    )


def check_remote_url(remote_url: str) -> None:
    """Belt-and-suspenders against pushing the publish to the wrong remote."""
    for pat in ALLOWED_REMOTE_PATTERNS:
        if pat.match(remote_url):
            return
    allowed = ", ".join(p.pattern for p in ALLOWED_REMOTE_PATTERNS)
    raise PublishError(
        f"REMOTE-URL GUARDRAIL TRIPPED: configured origin '{remote_url}' "
        f"does not match an allowed StemSplit/demucs-onnx URL.\n"
        f"  Allowed patterns: {allowed}\n"
        f"  Refusing to push. If you intentionally need to push elsewhere, "
        f"edit publish.py's ALLOWED_REMOTE_PATTERNS first."
    )


def check_file_count(repo_dir: Path, *, max_files: int) -> int:
    """Count files in the staged commit tree; abort if > ``max_files``.

    Uses the command the brief specified, with ``--root`` added so root
    (parent-less) commits are diffed against the empty tree instead of
    silently returning zero files:

        git diff-tree --no-commit-id --name-only -r --root HEAD | wc -l

    Without ``--root`` this returns 0 for the initial commit of an
    orphan publish repo, which would silently bypass the guardrail — the
    exact opposite of what we want.
    """
    result = _run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", "HEAD"],
        cwd=repo_dir,
    )
    files = [line for line in result.stdout.splitlines() if line.strip()]
    count = len(files)
    if count > max_files:
        preview = "\n    ".join(files[:20])
        more = f"\n    ... and {count - 20} more" if count > 20 else ""
        raise PublishError(
            f"FILE-COUNT GUARDRAIL TRIPPED: HEAD has {count} files, "
            f"max allowed is {max_files} (set via DEMUCS_ONNX_PUBLISH_MAX_FILES). "
            f"This is exactly the safety net that would have caught the "
            f"v0.3.3 monorepo-leak incident (2,638 files).\n"
            f"  Offending files (first 20):\n    {preview}{more}\n"
            f"  Refusing to push. Inspect what got staged before raising "
            f"the threshold."
        )
    return count


def resolve_max_files() -> int:
    raw = os.environ.get("DEMUCS_ONNX_PUBLISH_MAX_FILES")
    if raw is None:
        return DEFAULT_MAX_FILES
    try:
        value = int(raw)
    except ValueError as e:
        raise PublishError(
            f"DEMUCS_ONNX_PUBLISH_MAX_FILES must be an integer, got {raw!r}"
        ) from e
    if value < HARD_FLOOR_MAX_FILES:
        raise PublishError(
            f"DEMUCS_ONNX_PUBLISH_MAX_FILES={value} is below the hard floor "
            f"of {HARD_FLOOR_MAX_FILES}. The guardrail cannot be disabled."
        )
    return value


def is_live(args: argparse.Namespace) -> bool:
    if args.no_dry_run:
        return True
    if os.environ.get("PUBLISH_LIVE") == "1":
        return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Safe publish flow for the public StemSplit/demucs-onnx repo. "
            "Default is --dry-run (planning only, no push)."
        )
    )
    p.add_argument(
        "--pkg-dir",
        type=Path,
        default=PKG_DIR_DEFAULT,
        help=(
            "Path to the demucs-onnx package directory. Defaults to the "
            "parent of this script's directory (scripts/demucs-onnx-pkg/). "
            "Override only for testing."
        ),
    )
    p.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help=(
            "Override the publish remote URL. Default: "
            "https://github.com/StemSplit/demucs-onnx.git. Any override "
            "must still match the ALLOWED_REMOTE_PATTERNS allowlist."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Default. Plan + guardrails, no push, no gh release create.",
    )
    g.add_argument(
        "--no-dry-run",
        dest="no_dry_run",
        action="store_true",
        default=False,
        help="Actually push and create the GitHub release. Or set PUBLISH_LIVE=1.",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "Override the temp working directory. Default: a fresh "
            "mkdtemp() under the system temp dir. Useful for tests."
        ),
    )
    p.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Do not delete the temp workdir on success (debugging).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pkg_dir: Path = args.pkg_dir.resolve()
    if not pkg_dir.exists():
        print(f"ERROR: pkg-dir does not exist: {pkg_dir}", file=sys.stderr)
        return 2

    live = is_live(args)
    mode = "LIVE" if live else "DRY-RUN"
    print(f"=== demucs-onnx publish [{mode}] ===")
    print(f"  pkg_dir: {pkg_dir}")

    try:
        version = read_version_from_pyproject(pkg_dir)
    except PublishError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"  version: {version}")

    remote_url = args.remote_url or "https://github.com/StemSplit/demucs-onnx.git"
    try:
        check_remote_url(remote_url)
    except PublishError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    print(f"  remote_url: {remote_url} (passes allowlist)")

    try:
        max_files = resolve_max_files()
    except PublishError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    print(f"  max_files: {max_files}")

    changelog_subtitle, changelog_body = extract_changelog_for(pkg_dir, version)
    if not changelog_body:
        print(
            f"  WARNING: no CHANGELOG.md section found for version {version}. "
            f"Tag annotation + release body will be minimal."
        )

    if args.workdir is not None:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix=f"demucs-onnx-publish-{version}-"))
    print(f"  workdir: {workdir}")

    repo_dir = workdir / "repo"

    success = False
    try:
        print("\n--- Phase 1: copy subtree ---")
        copy_subtree(pkg_dir, repo_dir)

        present = sorted(p.relative_to(repo_dir) for p in repo_dir.rglob("*") if p.is_file())
        print(f"  copied {len(present)} files")
        for p in present[:5]:
            print(f"    {p}")
        if len(present) > 5:
            print(f"    ... and {len(present) - 5} more")

        forbidden_hits: list[str] = []
        for p in present:
            name = p.name
            if name.startswith(".env") or name.endswith(".pem") or name.endswith(".key"):
                forbidden_hits.append(str(p))
            if name.startswith(".tmp_"):
                forbidden_hits.append(str(p))
        if forbidden_hits:
            raise PublishError(
                f"FORBIDDEN-FILE GUARDRAIL TRIPPED: the copied subtree "
                f"contains files that must never be published: "
                f"{forbidden_hits!r}. Check the exclude list in publish.py."
            )

        print("\n--- Phase 2: init orphan repo + commit ---")
        init_orphan_repo(repo_dir)

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        subject = (
            f"release: v{version} — {changelog_subtitle}"
            if changelog_subtitle
            else f"release: v{version}"
        )
        body = changelog_body or (
            "Initial public release."
            if version == "0.1.0"
            else f"v{version} release."
        )
        commit_sha = build_commit(
            repo_dir,
            subject=subject,
            body=body,
            author_date_iso=ts,
        )
        print(f"  commit: {commit_sha}")

        msg = _run(["git", "cat-file", "-p", commit_sha], cwd=repo_dir).stdout
        for forbidden_trailer in ("Co-authored-by:", "Generated-by:", "Signed-off-by:"):
            if forbidden_trailer in msg:
                raise PublishError(
                    f"FORBIDDEN-TRAILER GUARDRAIL TRIPPED: commit message "
                    f"contains '{forbidden_trailer}'. The publish flow expects "
                    f"build_commit() (via git commit-tree plumbing) to bypass "
                    f"any wrapper-injected trailers."
                )

        print("\n--- Phase 3: tag ---")
        tag_name = f"v{version}"
        build_annotated_tag(
            repo_dir,
            name=tag_name,
            annotation_subject=subject,
            annotation_body=body,
            tagger_date_iso=ts,
            target=commit_sha,
        )
        print(f"  tag: {tag_name} -> {commit_sha}")

        print("\n--- Phase 4: file-count guardrail ---")
        count = check_file_count(repo_dir, max_files=max_files)
        print(f"  file count: {count} <= {max_files} (OK)")

        print("\n--- Phase 5: configure remote ---")
        _run(
            ["git", "remote", "add", "origin", remote_url],
            cwd=repo_dir,
        )
        configured = _run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_dir,
        ).stdout.strip()
        check_remote_url(configured)
        print(f"  origin: {configured} (passes allowlist)")

        print("\n--- Phase 6: push ---")
        if live:
            _run(["git", "push", "-u", "origin", "main"], cwd=repo_dir)
            _run(["git", "push", "origin", "--tags"], cwd=repo_dir)
            print(f"  pushed main + tag {tag_name}")
        else:
            print("  (dry-run) would: git push -u origin main")
            print(f"  (dry-run) would: git push origin --tags  (only {tag_name})")

        print("\n--- Phase 7: GitHub release ---")
        release_title = (
            f"{tag_name} — {changelog_subtitle}"
            if changelog_subtitle
            else tag_name
        )
        if live:
            body_path = workdir / "release-body.md"
            body_path.write_text(body + "\n")
            _run(
                [
                    "gh", "release", "create", tag_name,
                    "--repo", "StemSplit/demucs-onnx",
                    "--title", release_title,
                    "--notes-file", str(body_path),
                    "--target", "main",
                    "--latest",
                ],
                cwd=repo_dir,
            )
        else:
            print(f"  (dry-run) would: gh release create {tag_name} "
                  f"--repo StemSplit/demucs-onnx --title '{release_title}' "
                  f"--notes-file <body> --target main --latest")

        print(f"\n=== publish [{mode}] complete: {tag_name} @ {commit_sha} ===")
        success = True
        return 0

    except PublishError as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"\nABORT: subprocess failed: {e}", file=sys.stderr)
        return 3
    finally:
        if success and not args.keep_workdir and args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"  cleaned up {workdir}")
        elif args.keep_workdir or not success:
            print(f"  kept workdir for inspection: {workdir}")


if __name__ == "__main__":
    sys.exit(main())
