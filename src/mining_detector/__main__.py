"""Command-line interface: `python -m mining_detector <command>`.

Commands:
    demo    Run on built-in synthetic data (works offline, about a minute at most).
    live    Run on real Sentinel-2 data for a bounding box and two dates.
    fetch   Download and cache one Sentinel-2 scene, then print its size and cloud cover.

Run `python -m mining_detector <command> --help` for the options of each command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_BBOX, DEFAULT_MAX_CLOUD, SEARCH_WINDOW_DAYS

EXAMPLE_T0 = "2024-07-27"
EXAMPLE_T1 = "2025-07-27"


def _progress_printer(fraction: float, message: str) -> None:
    """Print progress in a simple, readable form."""
    print(f"[{int(fraction * 100):3d}%] {message}", flush=True)


def _print_result(result) -> None:
    """Print a short human-readable summary of a run."""
    s = result.summary
    print()
    print("Result")
    print(f"  New disturbance alerts : {s['n_alerts']}")
    print(f"  Total new area         : {s['total_new_disturbance_ha']} ha")
    for status, ha in s["by_legal_status"].items():
        print(f"    {status:<9}: {ha} ha")
    print(f"  alerts.geojson         : {result.alerts_path}")
    print(f"  report.json            : {result.report_path}")
    if result.metrics:
        print("  Metrics:")
        for k, v in result.metrics.items():
            print(f"    {k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with friendly help text."""
    parser = argparse.ArgumentParser(
        prog="python -m mining_detector",
        description=(
            "Find new surface-mining disturbance in Sentinel-2 satellite images and "
            "classify each site as legal, disputed or illegal against concession polygons."
        ),
        epilog="Start with:  python -m mining_detector demo",
    )
    parser.add_argument("--version", action="version", version=f"mining-detector {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="print detailed logs")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p_demo = sub.add_parser("demo", help="run on built-in synthetic data (offline)")
    p_demo.add_argument("--out", default="outputs", help="output folder (default: outputs)")

    p_live = sub.add_parser("live", help="run on real Sentinel-2 data")
    p_live.add_argument(
        "--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
        help="area of interest in decimal degrees: west south east north",
    )
    p_live.add_argument("--t0", help="earlier date, YYYY-MM-DD")
    p_live.add_argument("--t1", help="later date, YYYY-MM-DD")
    p_live.add_argument("--concessions", help="concession polygons (GeoJSON or GPKG)")
    p_live.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    p_live.add_argument(
        "--max-cloud", type=float, default=DEFAULT_MAX_CLOUD,
        help=f"maximum scene cloud cover in percent (default: {DEFAULT_MAX_CLOUD:g})",
    )
    p_live.add_argument(
        "--hard-cases", help="CSV of human-labelled points that override the model locally",
    )
    p_live.add_argument(
        "--eval-hard-cases", action="store_true",
        help="with --hard-cases: split the labels in half and measure the gain from training "
             "on one half (runs the pipeline twice)",
    )
    p_live.add_argument(
        "--validation", help="hand-labelled check points (CSV lon,lat,label or GeoJSON) to score against",
    )
    p_live.add_argument(
        "--review-chips", type=int, default=0, metavar="N",
        help="also export N stratified image chips and a labelling CSV for hard cases (default: 0)",
    )
    p_live.add_argument(
        "--max-aoi-km2", type=float, default=400.0,
        help="refuse areas larger than this many square km (default: 400)",
    )
    p_live.add_argument(
        "--example", action="store_true",
        help=f"run the default example area ({DEFAULT_BBOX}) for {EXAMPLE_T0} vs {EXAMPLE_T1}",
    )

    p_fetch = sub.add_parser("fetch", help="download and cache one scene (connectivity check)")
    p_fetch.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                         default=list(DEFAULT_BBOX), help="area of interest (default: example area)")
    p_fetch.add_argument("--date", default=EXAMPLE_T1, help="target date, YYYY-MM-DD")
    p_fetch.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    p_fetch.add_argument("--window-days", type=int, default=SEARCH_WINDOW_DAYS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "demo":
            from .pipeline import run_demo

            result = run_demo(out_dir=args.out, progress=_progress_printer)
            _print_result(result)
            return 0

        if args.command == "live":
            from .pipeline import default_t0_for, run_live

            if args.example:
                bbox, t0, t1 = DEFAULT_BBOX, EXAMPLE_T0, EXAMPLE_T1
            else:
                if not (args.bbox and args.t1):
                    parser.error("live needs --bbox and --t1 (or use --example).")
                bbox, t1 = tuple(args.bbox), args.t1
                # Default: the same calendar date one year earlier, to avoid seasonal confusion.
                t0 = args.t0 or default_t0_for(t1)
            if args.eval_hard_cases:
                if not args.hard_cases:
                    parser.error("--eval-hard-cases needs --hard-cases.")
                from .validation import holdout_gain

                gain = holdout_gain(
                    run_live, args.hard_cases, Path(args.out) / "hard_case_eval",
                    aoi_bbox=bbox, t0_date=t0, t1_date=t1, concessions_path=args.concessions,
                    progress=_progress_printer, max_cloud=args.max_cloud,
                    max_aoi_km2=args.max_aoi_km2,
                )
                print("\nHard-case hold-out evaluation")
                for k, v in gain.items():
                    print(f"  {k}: {v}")
                return 0
            result = run_live(
                aoi_bbox=bbox, t0_date=t0, t1_date=t1, out_dir=args.out,
                concessions_path=args.concessions, progress=_progress_printer,
                max_cloud=args.max_cloud, max_aoi_km2=args.max_aoi_km2,
                hard_cases_path=args.hard_cases, validation_path=args.validation,
                review_chips=args.review_chips,
            )
            _print_result(result)
            return 0

        if args.command == "fetch":
            from .pipeline import fetch_scene_report

            fetch_scene_report(
                bbox=tuple(args.bbox), date_str=args.date, max_cloud=args.max_cloud,
                window_days=args.window_days, progress=_progress_printer,
            )
            return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
