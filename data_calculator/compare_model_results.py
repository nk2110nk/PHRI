#!/usr/bin/env python3
"""Build MiPN vs Transformer domain comparison tables and plots.

The default aggregation matches the requested figure shape:

- rows: domains shared by MiPN/Transformer expert/general, plus Average
- series: MiPN_expert, MiPN_general, Transformer_expert, Transformer_general
- metrics: (a) utility, (b) number of steps, (c) step efficiency

Each domain/series pools all opponent pairs and case1 through case6 TSV rows.
The script also writes a case-level CSV so case1..case6 can be checked
individually.
"""

from __future__ import annotations

import argparse
import csv
import html
import statistics
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DOMAIN_ORDER = (
    "Laptop",
    "ItexvsCypress",
    "IS_BT_Acquisition",
    "Grocery",
    "thompson",
    "Car",
    "EnergySmall_A",
)
EXPECTED_CASES = tuple(f"case{i}" for i in range(1, 7))
EXPECTED_ROWS_PER_CASE = 100

SERIES = (
    ("MiPN_expert", "data_mipn", "expert"),
    ("MiPN_general", "data_mipn", "general"),
    ("Transformer_expert", "data_transformer", "expert"),
    ("Transformer_general", "data_transformer", "general"),
)
METRICS = (
    ("utility", "(a) Utility"),
    ("steps", "(b) Number of steps"),
    ("step_efficiency", "(c) Step efficiency"),
)
AXIS_LIMITS = {
    "utility": 1.0,
    "steps": 80.0,
    "step_efficiency": 0.35,
}
COLORS = {
    "MiPN_expert": "#d7191c",
    "MiPN_general": "#ffd700",
    "Transformer_expert": "#984ea3",
    "Transformer_general": "#008837",
}


@dataclass
class Stats:
    mean: float
    std: float
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MiPN/Transformer comparison CSVs and a 3-panel bar chart."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_comparison_outputs"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional domain list. Default: domains shared by all four series.",
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


def warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def ordered_names(names: set[str]) -> list[str]:
    known = [name for name in DOMAIN_ORDER if name in names]
    extras = sorted(name for name in names if name not in DOMAIN_ORDER)
    return known + extras


def discover_domains(strict: bool) -> list[str]:
    common_domains: set[str] | None = None
    for _, root, split in SERIES:
        split_dir = Path(root) / split
        if not split_dir.exists():
            warn_or_raise(f"missing split directory: {split_dir}", strict)
            continue

        domains: set[str] = set()
        for pair_dir in split_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            domains.update(path.name for path in pair_dir.iterdir() if path.is_dir())

        common_domains = domains if common_domains is None else common_domains & domains

    if not common_domains:
        warn_or_raise("no shared domains found across all four series", strict)
        return []
    return ordered_names(common_domains)


def case_sort_key(case_dir: Path) -> tuple[int, str]:
    name = case_dir.name
    if name.startswith("case") and name[4:].isdigit():
        return (int(name[4:]), name)
    return (10**9, name)


def values_from_row(row: dict[str, str], tsv_file: Path, strict: bool) -> dict[str, float] | None:
    try:
        utility = float(row["my_util"])
        steps = float(row["step"])
    except KeyError as exc:
        warn_or_raise(f"{tsv_file}: missing column {exc.args[0]!r}", strict)
        return None
    except (TypeError, ValueError):
        warn_or_raise(f"{tsv_file}: non-numeric my_util or step value", strict)
        return None

    efficiency = utility / steps if steps != 0.0 else 0.0
    return {
        "utility": utility,
        "steps": steps,
        "step_efficiency": efficiency,
    }


def collect_values(
    root: Path,
    split: str,
    domain: str,
    strict: bool,
) -> dict[str, dict[str, list[float]]]:
    by_case: dict[str, dict[str, list[float]]] = {
        case_name: {metric: [] for metric, _ in METRICS} for case_name in EXPECTED_CASES
    }
    split_dir = root / split
    pair_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())

    for pair_dir in pair_dirs:
        domain_dir = pair_dir / domain
        if not domain_dir.exists():
            warn_or_raise(f"missing domain directory: {domain_dir}", strict)
            continue

        case_dirs = sorted(
            [
                path
                for path in domain_dir.iterdir()
                if path.is_dir() and path.name.startswith("case")
            ],
            key=case_sort_key,
        )
        case_names = tuple(path.name for path in case_dirs)
        if case_names != EXPECTED_CASES:
            warn_or_raise(
                f"{root.name}/{split}/{pair_dir.name}/{domain}: "
                f"expected {EXPECTED_CASES}, found {case_names}",
                strict,
            )

        for case_dir in case_dirs:
            if case_dir.name not in by_case:
                continue

            tsv_files = sorted(case_dir.glob("*.tsv"))
            if len(tsv_files) != 1:
                warn_or_raise(
                    f"{root.name}/{split}/{pair_dir.name}/{domain}/{case_dir.name}: "
                    f"expected 1 TSV, found {len(tsv_files)}",
                    strict,
                )

            row_count = 0
            for tsv_file in tsv_files:
                with tsv_file.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        row_count += 1
                        values = values_from_row(row, tsv_file, strict)
                        if values is None:
                            continue
                        for metric, value in values.items():
                            by_case[case_dir.name][metric].append(value)

            if row_count != EXPECTED_ROWS_PER_CASE:
                warn_or_raise(
                    f"{root.name}/{split}/{pair_dir.name}/{domain}/{case_dir.name}: "
                    f"expected {EXPECTED_ROWS_PER_CASE} rows, found {row_count}",
                    strict,
                )

    return by_case


def make_stats(values: list[float]) -> Stats:
    if not values:
        return Stats(mean=0.0, std=0.0, count=0)
    return Stats(
        mean=statistics.fmean(values),
        std=statistics.pstdev(values) if len(values) > 1 else 0.0,
        count=len(values),
    )


def aggregate(domains: list[str], strict: bool) -> tuple[dict, dict]:
    summary: dict[str, dict[str, dict[str, Stats]]] = defaultdict(lambda: defaultdict(dict))
    case_summary: dict[str, dict[str, dict[str, dict[str, Stats]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    average_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for series_name, root_name, split in SERIES:
        for domain in domains:
            by_case = collect_values(Path(root_name), split, domain, strict)
            pooled: dict[str, list[float]] = {metric: [] for metric, _ in METRICS}

            for case_name in EXPECTED_CASES:
                for metric, _ in METRICS:
                    values = by_case[case_name][metric]
                    pooled[metric].extend(values)
                    case_summary[metric][domain][series_name][case_name] = make_stats(values)

            for metric, _ in METRICS:
                summary[metric][domain][series_name] = make_stats(pooled[metric])
                average_values[metric][series_name].extend(pooled[metric])

    for metric, _ in METRICS:
        for series_name, _, _ in SERIES:
            summary[metric]["Average"][series_name] = make_stats(
                average_values[metric][series_name]
            )

    return summary, case_summary


def write_summary_csv(path: Path, summary: dict, domains: list[str]) -> None:
    rows = [["metric", "domain", "series", "mean", "std", "count"]]
    for metric, _ in METRICS:
        for domain in [*domains, "Average"]:
            for series_name, _, _ in SERIES:
                stats = summary[metric][domain][series_name]
                rows.append(
                    [
                        metric,
                        domain,
                        series_name,
                        f"{stats.mean:.6f}",
                        f"{stats.std:.6f}",
                        str(stats.count),
                    ]
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def write_case_csv(path: Path, case_summary: dict, domains: list[str]) -> None:
    rows = [["metric", "domain", "series", "case", "mean", "std", "count"]]
    for metric, _ in METRICS:
        for domain in domains:
            for series_name, _, _ in SERIES:
                for case_name in EXPECTED_CASES:
                    stats = case_summary[metric][domain][series_name][case_name]
                    rows.append(
                        [
                            metric,
                            domain,
                            series_name,
                            case_name,
                            f"{stats.mean:.6f}",
                            f"{stats.std:.6f}",
                            str(stats.count),
                        ]
                    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def nice_axis_max(value: float) -> float:
    if value <= 0:
        return 1.0
    candidates = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    for candidate in candidates:
        if value <= candidate:
            return candidate
    magnitude = 10 ** (len(str(int(value))) - 1)
    return ((int(value / magnitude) + 1) * magnitude)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#333">{html.escape(text)}</text>'
    )


def plot_summary_svg(path: Path, summary: dict, domains: list[str]) -> None:
    plot_domains = [*domains, "Average"]
    width = 1740
    height = max(720, 90 * len(plot_domains))
    panel_width = 430
    panel_gap = 50
    left_margin = 135
    top_margin = 40
    bottom_margin = 70
    plot_height = height - top_margin - bottom_margin
    row_gap = plot_height / len(plot_domains)
    bar_height = 13
    offsets = (-22.5, -7.5, 7.5, 22.5)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for panel_index, (metric, label) in enumerate(METRICS):
        panel_x = left_margin + panel_index * (panel_width + panel_gap)
        max_value = max(
            summary[metric][domain][series_name].mean
            + summary[metric][domain][series_name].std
            for domain in plot_domains
            for series_name, _, _ in SERIES
        )
        axis_max = AXIS_LIMITS.get(metric, nice_axis_max(max_value * 1.05))
        axis_y = top_margin + plot_height

        parts.append(
            f'<rect x="{panel_x:.1f}" y="{top_margin:.1f}" '
            f'width="{panel_width:.1f}" height="{plot_height:.1f}" '
            f'fill="none" stroke="#333" stroke-width="1"/>'
        )

        tick_count = 5
        for tick in range(tick_count + 1):
            value = axis_max * tick / tick_count
            x = panel_x + panel_width * value / axis_max
            parts.append(
                f'<line x1="{x:.1f}" y1="{axis_y:.1f}" x2="{x:.1f}" '
                f'y2="{axis_y + 5:.1f}" stroke="#333" stroke-width="1"/>'
            )
            if metric == "step_efficiency":
                tick_label = f"{value:.2f}"
            else:
                tick_label = f"{value:.1f}" if axis_max <= 2 else f"{value:.0f}"
            parts.append(svg_text(x, axis_y + 22, tick_label, size=11, anchor="middle"))

        for row_index, domain in enumerate(plot_domains):
            row_center = top_margin + row_gap * (row_index + 0.5)
            if panel_index == 0:
                parts.append(svg_text(panel_x - 10, row_center + 4, domain, size=12, anchor="end"))

            for offset, (series_name, _, _) in zip(offsets, SERIES):
                stats = summary[metric][domain][series_name]
                bar_y = row_center + offset - bar_height / 2
                bar_w = panel_width * stats.mean / axis_max
                err_low = max(0.0, stats.mean - stats.std)
                err_high = min(axis_max, stats.mean + stats.std)
                err_x1 = panel_x + panel_width * err_low / axis_max
                err_x2 = panel_x + panel_width * err_high / axis_max
                err_y = row_center + offset

                parts.append(
                    f'<rect x="{panel_x:.1f}" y="{bar_y:.1f}" '
                    f'width="{bar_w:.1f}" height="{bar_height:.1f}" '
                    f'fill="{COLORS[series_name]}"/>'
                )
                parts.append(
                    f'<line x1="{err_x1:.1f}" y1="{err_y:.1f}" '
                    f'x2="{err_x2:.1f}" y2="{err_y:.1f}" '
                    f'stroke="#888" stroke-width="1.5"/>'
                )

        parts.append(svg_text(panel_x + panel_width / 2, height - 16, label, size=13, anchor="middle"))

    legend_x = left_margin + 3 * (panel_width + panel_gap) - 10
    legend_y = height - 165
    parts.append(
        f'<rect x="{legend_x - 10:.1f}" y="{legend_y - 22:.1f}" '
        f'width="160" height="108" fill="white" stroke="#ccc" stroke-width="1"/>'
    )
    for index, (series_name, _, _) in enumerate(SERIES):
        y = legend_y + index * 24
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{y - 10:.1f}" width="24" height="10" '
            f'fill="{COLORS[series_name]}"/>'
        )
        parts.append(svg_text(legend_x + 32, y, series_name, size=12))

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_summary_matplotlib(path: Path, summary: dict, domains: list[str], dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    plot_domains = [*domains, "Average"]
    y_positions = list(range(len(plot_domains)))
    bar_height = 0.16
    offsets = (-1.5 * bar_height, -0.5 * bar_height, 0.5 * bar_height, 1.5 * bar_height)
    figure_height = max(7.0, 0.82 * len(plot_domains))

    fig, axes = plt.subplots(1, 3, figsize=(15, figure_height), sharey=True)

    for axis, (metric, label) in zip(axes, METRICS):
        for offset, (series_name, _, _) in zip(offsets, SERIES):
            means = [summary[metric][domain][series_name].mean for domain in plot_domains]
            stds = [summary[metric][domain][series_name].std for domain in plot_domains]
            axis.barh(
                [y + offset for y in y_positions],
                means,
                height=bar_height,
                color=COLORS[series_name],
                label=series_name,
                xerr=stds,
                error_kw={"ecolor": "0.55", "elinewidth": 1.0, "capsize": 0},
            )

        axis.set_xlabel(label)
        if metric in AXIS_LIMITS:
            axis.set_xlim(0, AXIS_LIMITS[metric])
        if metric == "step_efficiency":
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.grid(False)
        axis.tick_params(axis="both", labelsize=8)
        axis.set_ylim(-0.8, len(plot_domains) - 0.2)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(plot_domains)
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.6, rect=(0.0, 0.0, 0.86, 1.0))
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.87, 0.13),
        fontsize=8,
        frameon=True,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains if args.domains is not None else discover_domains(args.strict)
    summary, case_summary = aggregate(domains, args.strict)

    summary_csv = args.output_dir / "model_comparison_summary.csv"
    case_csv = args.output_dir / "model_comparison_case_summary.csv"
    svg_path = args.output_dir / "model_comparison.svg"
    png_path = args.output_dir / "model_comparison.png"
    pdf_path = args.output_dir / "model_comparison.pdf"

    write_summary_csv(summary_csv, summary, domains)
    write_case_csv(case_csv, case_summary, domains)
    plot_summary_svg(svg_path, summary, domains)
    try:
        plot_summary_matplotlib(png_path, summary, domains, args.dpi)
        plot_summary_matplotlib(pdf_path, summary, domains, args.dpi)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipped PNG/PDF outputs")

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved case CSV: {case_csv}")
    print(f"Saved figure SVG: {svg_path}")
    if png_path.exists():
        print(f"Saved figure PNG: {png_path}")
    if pdf_path.exists():
        print(f"Saved figure PDF: {pdf_path}")


if __name__ == "__main__":
    main()
