#!/usr/bin/env python3
"""Create a separate model comparison figure grouped by opponent pair.

This is the "d" counterpart to compare_model_results.py.  It keeps the same
four model series and the same three metrics, but the y-axis is opponent-pair
combinations.  Domains are pooled together, and case1 through case6 are all
included.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from compare_model_results import (
    DOMAIN_ORDER,
    EXPECTED_CASES,
    EXPECTED_ROWS_PER_CASE,
    METRICS,
    SERIES,
    Stats,
    make_stats,
    plot_summary_matplotlib,
    plot_summary_svg,
    values_from_row,
    warn_or_raise,
)


AGENT_ORDER = ("Boulware", "Conceder", "Linear", "Atlas3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MiPN/Transformer comparison outputs grouped by opponent pair."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_comparison_outputs"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing cases, missing TSVs, or unexpected row counts.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output DPI. Default: 300",
    )
    return parser.parse_args()


def pair_sort_key(pair: str) -> tuple[int, int, str]:
    left, sep, right = pair.partition("-")
    if not sep:
        return (len(AGENT_ORDER), len(AGENT_ORDER), pair)
    left_index = AGENT_ORDER.index(left) if left in AGENT_ORDER else len(AGENT_ORDER)
    right_index = AGENT_ORDER.index(right) if right in AGENT_ORDER else len(AGENT_ORDER)
    return (left_index, right_index, pair)


def discover_pairs(strict: bool) -> list[str]:
    common_pairs: set[str] | None = None
    for _, root, split in SERIES:
        split_dir = Path(root) / split
        if not split_dir.exists():
            warn_or_raise(f"missing split directory: {split_dir}", strict)
            continue
        pairs = {path.name for path in split_dir.iterdir() if path.is_dir()}
        common_pairs = pairs if common_pairs is None else common_pairs & pairs

    if not common_pairs:
        warn_or_raise("no shared opponent pairs found across all four series", strict)
        return []
    return sorted(common_pairs, key=pair_sort_key)


def collect_pair_values(
    root: Path,
    split: str,
    pair: str,
    strict: bool,
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {metric: [] for metric, _ in METRICS}
    pair_dir = root / split / pair
    if not pair_dir.exists():
        warn_or_raise(f"missing pair directory: {pair_dir}", strict)
        return values

    for domain in DOMAIN_ORDER:
        domain_dir = pair_dir / domain
        if not domain_dir.exists():
            warn_or_raise(f"missing domain directory: {domain_dir}", strict)
            continue

        for case_name in EXPECTED_CASES:
            case_dir = domain_dir / case_name
            tsv_files = sorted(case_dir.glob("*.tsv"))
            if len(tsv_files) != 1:
                warn_or_raise(
                    f"{root.name}/{split}/{pair}/{domain}/{case_name}: "
                    f"expected 1 TSV, found {len(tsv_files)}",
                    strict,
                )

            row_count = 0
            for tsv_file in tsv_files:
                with tsv_file.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        row_count += 1
                        row_values = values_from_row(row, tsv_file, strict)
                        if row_values is None:
                            continue
                        for metric, value in row_values.items():
                            values[metric].append(value)

            if row_count != EXPECTED_ROWS_PER_CASE:
                warn_or_raise(
                    f"{root.name}/{split}/{pair}/{domain}/{case_name}: "
                    f"expected {EXPECTED_ROWS_PER_CASE} rows, found {row_count}",
                    strict,
                )

    return values


def aggregate(pairs: list[str], strict: bool) -> dict[str, dict[str, dict[str, Stats]]]:
    summary: dict[str, dict[str, dict[str, Stats]]] = defaultdict(lambda: defaultdict(dict))
    average_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for series_name, root_name, split in SERIES:
        for pair in pairs:
            pair_values = collect_pair_values(Path(root_name), split, pair, strict)
            for metric, _ in METRICS:
                values = pair_values[metric]
                summary[metric][pair][series_name] = make_stats(values)
                average_values[metric][series_name].extend(values)

    for metric, _ in METRICS:
        for series_name, _, _ in SERIES:
            summary[metric]["Average"][series_name] = make_stats(
                average_values[metric][series_name]
            )

    return summary


def write_summary_csv(path: Path, summary: dict, pairs: list[str]) -> None:
    rows = [["metric", "opponent_pair", "series", "mean", "std", "count"]]
    for metric, _ in METRICS:
        for pair in [*pairs, "Average"]:
            for series_name, _, _ in SERIES:
                stats = summary[metric][pair][series_name]
                rows.append(
                    [
                        metric,
                        pair,
                        series_name,
                        f"{stats.mean:.6f}",
                        f"{stats.std:.6f}",
                        str(stats.count),
                    ]
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.strict)
    summary = aggregate(pairs, args.strict)

    csv_path = args.output_dir / "model_comparison_by_pair_summary.csv"
    svg_path = args.output_dir / "model_comparison_by_pair.svg"
    png_path = args.output_dir / "model_comparison_by_pair.png"
    pdf_path = args.output_dir / "model_comparison_by_pair.pdf"

    write_summary_csv(csv_path, summary, pairs)
    plot_summary_svg(svg_path, summary, pairs)
    try:
        plot_summary_matplotlib(png_path, summary, pairs, args.dpi)
        plot_summary_matplotlib(pdf_path, summary, pairs, args.dpi)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipped PNG/PDF outputs")

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved figure SVG: {svg_path}")
    if png_path.exists():
        print(f"Saved figure PNG: {png_path}")
    if pdf_path.exists():
        print(f"Saved figure PDF: {pdf_path}")


if __name__ == "__main__":
    main()
