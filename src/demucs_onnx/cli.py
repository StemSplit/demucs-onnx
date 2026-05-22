"""Command-line interface for ``demucs-onnx``.

::

    # 4 stems, auto execution provider, fp32 weights, .wav output
    demucs-onnx separate song.mp3 out/

    # the killer feature — instant karaoke instrumental
    demucs-onnx separate song.mp3 out/ --karaoke --mp3

    # just the vocals, smaller download, GPU EP picked automatically
    demucs-onnx separate song.mp3 out/ --stem vocals --small

    # 6-stem variant (4 + guitar + piano), single-file flavor (new in 0.3)
    demucs-onnx separate song.mp3 out/ --model htdemucs_6s --stems guitar piano

    # write a copy-pasteable bundler config snippet for Vite
    demucs-onnx browser-config --bundler vite

    # scaffold a zero-build vanilla HTML/JS browser demo
    demucs-onnx browser-demo /tmp/browser_demo

    # export your own checkpoint
    demucs-onnx export htdemucs_ft out/

    # see every supported model + variant
    demucs-onnx list-models
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .inference import (
    ALL_KNOWN_STEMS,
    DEFAULT_BAG_MODEL,
    list_models,
    separate,
)

PROVIDER_CHOICES = ["auto", "cpu", "coreml", "cuda", "dml", "wasm"]
BUNDLER_CHOICES = ["vite", "webpack", "esbuild", "next", "rollup"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demucs-onnx",
        description=(
            "Run and export HT-Demucs / Demucs music source separation as "
            "ONNX. Pure numpy + onnxruntime for inference."
        ),
    )
    p.add_argument("--version", action="version", version=f"demucs-onnx {__version__}")

    sub = p.add_subparsers(dest="command", required=True)

    # ---- separate -----------------------------------------------------------
    sep = sub.add_parser(
        "separate",
        help="Run separation on an audio file (auto-downloads model from HF).",
    )
    sep.add_argument("input", type=Path, help="Input audio file. Any sample rate.")
    sep.add_argument("output_dir", type=Path,
                     help="Directory to write per-stem output files into.")
    sep.add_argument("--model", default=DEFAULT_BAG_MODEL,
                     help=(
                         "Model name. Defaults to 'htdemucs_ft' (4-stem "
                         "specialist bag). Other choices: 'htdemucs' "
                         "(single-file 4-stem, ~30%% faster, slightly lower "
                         "SDR), 'htdemucs_6s' (single-file 6-stem with "
                         "guitar + piano), or any specialist alias "
                         "('drums'/'bass'/'other'/'vocals'/'ft_drums'/...)."
                     ))
    sep.add_argument("--stem", default=None,
                     help=(
                         "Shortcut: run only this one stem. For "
                         "drums/bass/other/vocals: equivalent to "
                         "`--model htdemucs_ft_<stem>` (~4x faster than the "
                         "bag). For guitar/piano: routes to htdemucs_6s."
                     ))
    sep.add_argument("--stems", nargs="+", default=None,
                     choices=list(ALL_KNOWN_STEMS),
                     help=(
                         "Subset of stems to compute. Saves time on the "
                         "FT bag by skipping specialists you don't need. "
                         "On htdemucs/htdemucs_6s the model always "
                         "computes every stem internally; this just "
                         "filters the output dict."
                     ))
    sep.add_argument("--providers", default="auto", choices=PROVIDER_CHOICES,
                     dest="providers",
                     help=(
                         "ONNX Runtime execution providers. Default: 'auto' "
                         "(CoreML on macOS, CUDA on Linux+NVIDIA, DML on "
                         "Windows, CPU otherwise)."
                     ))
    sep.add_argument("--provider", default=None, dest="provider_legacy",
                     help=argparse.SUPPRESS)
    sep.add_argument("--precision", default="fp32",
                     choices=["fp32", "fp16weights"],
                     help=(
                         "Weight storage precision. fp16weights downloads "
                         "1.91x smaller files with no runtime cost (max diff "
                         "~6e-5 vs fp32). Default: fp32."
                     ))
    sep.add_argument("--small", action="store_true",
                     help="Alias for --precision fp16weights.")
    sep.add_argument("--mp3", action="store_true",
                     help=(
                         "Write .mp3 instead of .wav. Requires the 'mp3' "
                         "extra: pip install 'demucs-onnx[mp3]'."
                     ))
    sep.add_argument("--bitrate", default="192k",
                     help="MP3 bitrate (e.g. 128k, 192k, 256k, 320k). Default: 192k.")
    sep.add_argument("--mix-stems", default=None,
                     help=(
                         "Comma-separated list of stems to sum into one "
                         "extra output file. Example: --mix-stems vocals,drums. "
                         "On htdemucs_6s you can mix guitar,piano too."
                     ))
    sep.add_argument("--karaoke", action="store_true",
                     help=(
                         "Killer-feature shortcut: sum drums+bass+other into "
                         "one karaoke instrumental track (= --mix-stems "
                         "drums,bass,other). Use with --mp3 to get a "
                         "ready-to-share karaoke.mp3."
                     ))
    sep.add_argument("--cache-dir", type=Path, default=None,
                     help="Override the huggingface_hub model cache.")
    sep.add_argument("-q", "--quiet", action="store_true",
                     help="Suppress all output except errors.")
    sep.add_argument("-v", "--verbose", action="store_true",
                     help="Print chunk-by-chunk progress instead of a bar.")

    # ---- export -------------------------------------------------------------
    exp = sub.add_parser(
        "export",
        help="Convert a demucs/htdemucs PyTorch checkpoint to ONNX.",
    )
    exp.add_argument("checkpoint",
                     help=(
                         "demucs.pretrained name (e.g. htdemucs_ft, "
                         "htdemucs, htdemucs_6s) or a path to a local .th "
                         "checkpoint."
                     ))
    exp.add_argument("output", type=Path,
                     help=(
                         "Output path. For a single stem: a .onnx filename. "
                         "For the FT bag with multiple stems: a directory."
                     ))
    exp.add_argument("--stem", default=None, choices=list(ALL_KNOWN_STEMS),
                     help="Single stem to export from the FT bag.")
    exp.add_argument("--stems", nargs="+", default=None,
                     choices=list(ALL_KNOWN_STEMS),
                     help="Subset of stems to export from the FT bag.")
    exp.add_argument("--opset", type=int, default=17,
                     help="ONNX opset version. Default: 17.")
    exp.add_argument("--no-parity-check", action="store_true",
                     help=(
                         "Skip the numerical parity check. Not recommended; "
                         "this is the safety net that catches broken exports."
                     ))
    exp.add_argument("--parity-tolerance", type=float, default=1e-3,
                     help="Max allowed abs diff in parity check. Default: 1e-3.")
    exp.add_argument("-q", "--quiet", action="store_true",
                     help="Suppress progress output.")

    # ---- list-models --------------------------------------------------------
    sub.add_parser("list-models",
                   help="Print the supported model aliases and HF repo URLs.")

    # ---- browser-config -----------------------------------------------------
    bc = sub.add_parser(
        "browser-config",
        help=(
            "Print a copy-pasteable onnxruntime-web bundler config snippet "
            "for the given bundler (vite/webpack/esbuild/next/rollup)."
        ),
    )
    bc.add_argument("--bundler", default="vite", choices=BUNDLER_CHOICES,
                    help="Which bundler to generate the snippet for. Default: vite.")

    # ---- browser-demo -------------------------------------------------------
    bd = sub.add_parser(
        "browser-demo",
        help=(
            "Scaffold the browser demo files into the given directory so "
            "you can run `python -m http.server` and try it locally."
        ),
    )
    bd.add_argument("target", type=Path,
                    help="Empty (or new) directory to write demo files into.")
    bd.add_argument("--react", action="store_true",
                    help="Emit a Vite + React + TS project instead of the "
                         "zero-build vanilla HTML/JS demo.")
    bd.add_argument("--model-url", default=None,
                    help=(
                        "Override the ONNX model URL the demo loads. "
                        "Default: htdemucs-ft-vocals fp16weights from the "
                        "StemSplitio HF org."
                    ))

    # ---- prewarm ------------------------------------------------------------
    pw = sub.add_parser(
        "prewarm",
        help=(
            "Pre-download + pre-compile sessions for one or more models. "
            "Useful in CI / server-warmup contexts."
        ),
    )
    pw.add_argument("--models", nargs="+", default=None,
                    help="Models to prewarm. Default: htdemucs_ft (4 specialists).")
    pw.add_argument("--precision", default="fp32",
                    choices=["fp32", "fp16weights"])
    pw.add_argument("--providers", default="auto", choices=PROVIDER_CHOICES)
    pw.add_argument("--cache-dir", type=Path, default=None)

    return p


def _parse_bitrate(spec: str) -> int:
    """Parse '192k' / '192' / '192kbps' / '192 kbps' into an int kbps."""
    s = spec.strip().lower().replace(" ", "").rstrip("bps").rstrip("k")
    try:
        kbps = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --bitrate {spec!r} (expected e.g. '192k' or '192')",
        ) from exc
    if not 32 <= kbps <= 320:
        raise argparse.ArgumentTypeError(
            f"--bitrate {kbps} kbps out of range (32-320)",
        )
    return kbps


def _configure_logging(*, quiet: bool, verbose: bool) -> None:
    """Wire up sane logging defaults for the CLI.

    - In ``--quiet`` mode we keep only ERROR-level output and suppress
      ``huggingface_hub``'s progress bars / HTTP debug lines completely.
    - In ``--verbose`` mode demucs-onnx prints chunk-by-chunk INFO logs.
    - In the default mode (neither flag) we still suppress hf-hub's
      noisy INFO HTTP logs but keep its progress bars (useful on first
      download).
    """
    if quiet:
        logging.getLogger("demucs_onnx").setLevel(logging.ERROR)
    else:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)

    if not verbose:
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("hf_transfer").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    if quiet:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except ImportError:
            pass


def cmd_separate(args: argparse.Namespace) -> int:
    _configure_logging(quiet=args.quiet, verbose=args.verbose)

    model = args.model
    if args.stem is not None:
        if args.stem in ("drums", "bass", "other", "vocals"):
            model = f"htdemucs_ft_{args.stem}"
        elif args.stem in ("guitar", "piano"):
            model = "htdemucs_6s"
            args.stems = [args.stem]
        else:
            print(f"error: unknown stem {args.stem!r}", file=sys.stderr)
            return 2

    precision = "fp16weights" if args.small else args.precision

    providers = args.providers
    if args.provider_legacy is not None:
        providers = args.provider_legacy

    mix_stems: list[str] | None = None
    mix_name = "mix"
    if args.karaoke:
        mix_stems = ["drums", "bass", "other"]
        mix_name = "karaoke"
    elif args.mix_stems:
        mix_stems = [s.strip() for s in args.mix_stems.split(",") if s.strip()]
        mix_name = "mix"

    output_format = "mp3" if args.mp3 else "wav"
    try:
        bitrate_kbps = _parse_bitrate(args.bitrate)
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    separate(
        input=args.input,
        output_dir=args.output_dir,
        model=model,
        stems=args.stems,
        providers=providers,
        precision=precision,
        cache_dir=args.cache_dir,
        verbose=args.verbose and not args.quiet,
        progress=not args.quiet,
        output_format=output_format,
        bitrate_kbps=bitrate_kbps,
        mix_stems=mix_stems,
        mix_output_name=mix_name,
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    try:
        from .export import export_to_onnx
    except ImportError as exc:
        print(
            "error: the export feature requires the 'export' extra. "
            "Install with: pip install 'demucs-onnx[export]'\n"
            f"underlying ImportError: {exc}",
            file=sys.stderr,
        )
        return 3

    out_paths = export_to_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        stem=args.stem,
        stems=args.stems,
        opset=args.opset,
        parity_check=not args.no_parity_check,
        parity_tolerance=args.parity_tolerance,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print("\nExported:")
        for stem, path in out_paths.items():
            print(f"  {stem:8s} -> {path}")
    return 0


def cmd_list_models(_args: argparse.Namespace) -> int:
    models = list_models()
    print(f"{'alias':<22}  {'kind':<14}  {'sources':<32}  {'precision':<12}  url")
    print(f"{'-' * 22}  {'-' * 14}  {'-' * 32}  {'-' * 12}  {'-' * 50}")
    for alias in sorted(models):
        m = models[alias]
        kind = m.get("kind", "specialist")
        sources = m.get("sources", "")
        for precision in ("fp32", "fp16weights"):
            url = m.get(precision, m["repo"])
            print(f"{alias:<22}  {kind:<14}  {sources:<32}  {precision:<12}  {url}")
    return 0


def cmd_browser_config(args: argparse.Namespace) -> int:
    from .browser import print_wasm_config

    print_wasm_config(args.bundler)
    return 0


def cmd_browser_demo(args: argparse.Namespace) -> int:
    from .browser import DEFAULT_DEMO_URL, write_demo_dir

    try:
        files = write_demo_dir(args.target, model_url=args.model_url, react=args.react)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(files)} files to {args.target}:")
    for f in files:
        print(f"  {f.relative_to(args.target.parent if args.target.parent.exists() else f.parent)}")
    print()
    print(f"Default model URL: {args.model_url or DEFAULT_DEMO_URL}")
    print()
    if args.react:
        print("To run:")
        print(f"  cd {args.target}")
        print("  npm install")
        print("  npm run dev")
    else:
        print("To run:")
        print(f"  cd {args.target}")
        print("  python -m http.server 8080")
        print("  open http://localhost:8080/")
    return 0


def cmd_prewarm(args: argparse.Namespace) -> int:
    from . import prewarm

    _configure_logging(quiet=False, verbose=False)
    paths = prewarm(
        models=args.models,
        precision=args.precision,
        providers=args.providers,
        cache_dir=args.cache_dir,
    )
    print(f"Prewarmed {len(paths)} model file(s):")
    for k, v in paths.items():
        print(f"  {k:14s} -> {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "separate":
        return cmd_separate(args)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "list-models":
        return cmd_list_models(args)
    if args.command == "browser-config":
        return cmd_browser_config(args)
    if args.command == "browser-demo":
        return cmd_browser_demo(args)
    if args.command == "prewarm":
        return cmd_prewarm(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
