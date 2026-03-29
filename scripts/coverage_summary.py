#!/usr/bin/env python3

import json
import sys
from pathlib import Path


UTILITY_CRATE_PATH_FRAGMENTS = (
    "/bench_report/src/",
    "/xtask/src/",
)
METRICS = ("regions", "functions", "lines")


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        payload = json.load(handle)
    return payload["data"][0]


def aggregate(files: list[dict], *, exclude_utility: bool) -> dict:
    totals = {metric: {"count": 0, "covered": 0} for metric in METRICS}
    for file_entry in files:
        filename = file_entry["filename"]
        if exclude_utility and any(
            fragment in filename for fragment in UTILITY_CRATE_PATH_FRAGMENTS
        ):
            continue
        summary = file_entry["summary"]
        for metric in METRICS:
            totals[metric]["count"] += summary[metric]["count"]
            totals[metric]["covered"] += summary[metric]["covered"]
    for metric in METRICS:
        count = totals[metric]["count"]
        covered = totals[metric]["covered"]
        totals[metric]["percent"] = 0.0 if count == 0 else 100.0 * covered / count
    return totals


def print_group(title: str, totals: dict) -> None:
    print(title)
    for metric in METRICS:
        summary = totals[metric]
        print(
            f"  {metric:>9}: {summary['percent']:6.2f}% "
            f"({summary['covered']}/{summary['count']})"
        )
    print()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: coverage_summary.py <coverage-summary.json>", file=sys.stderr)
        return 2

    summary = load_summary(Path(sys.argv[1]))
    files = summary["files"]
    print_group("Raw workspace coverage:", aggregate(files, exclude_utility=False))
    print_group(
        "Normalized coverage excluding utility crates (bench_report, xtask):",
        aggregate(files, exclude_utility=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
