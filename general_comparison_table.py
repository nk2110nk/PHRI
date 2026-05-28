#!/usr/bin/env python3
"""Create the general-policy domain table for MiPN vs Transformer.

The table keeps the fixed Domain and |Omega| labels from the thesis table, drops
Opposition, and aggregates MiPN_general / Transformer_general over all opponent
pairs and case1 through case6.
"""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path


EXPECTED_CASES = tuple(f"case{i}" for i in range(1, 7))
EXPECTED_ROWS_PER_CASE = 100

SERIES = (
    ("MiPN_general", Path("data_mipn") / "general"),
    ("Transformer_general", Path("data_transformer") / "general"),
)

DOMAIN_META = (
    ("Manufacturing", "Laptop", "27"),
    ("SCM", "ItexvsCypress", "180"),
    ("Buyout", "IS_BT_Acquisition", "384"),
    ("Retail", "Grocery", "1600"),
    ("Employment", "thompson", "3125"),
    ("Mobility", "Car", "15625"),
    ("Energy Plant", "EnergySmall_A", "15625"),
)
TOP_DOMAINS = DOMAIN_META[:4]
BOTTOM_DOMAINS = DOMAIN_META[4:]


@dataclass
class DomainStats:
    utility: float
    agreement_rate: float
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MiPN_general vs Transformer_general table outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_comparison_outputs"),
        help="Directory for table outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing cases, missing TSVs, or unexpected row counts.",
    )
    return parser.parse_args()


def warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    print(f"warning: {message}")


def collect_domain_stats(split_dir: Path, domain: str, strict: bool) -> DomainStats:
    utilities: list[float] = []
    agreements = 0

    pair_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for pair_dir in pair_dirs:
        domain_dir = pair_dir / domain
        if not domain_dir.exists():
            warn_or_raise(f"missing domain directory: {domain_dir}", strict)
            continue

        case_dirs = sorted(path for path in domain_dir.iterdir() if path.is_dir())
        case_names = tuple(path.name for path in case_dirs if path.name.startswith("case"))
        if case_names != EXPECTED_CASES:
            warn_or_raise(
                f"{split_dir}/{pair_dir.name}/{domain}: expected {EXPECTED_CASES}, "
                f"found {case_names}",
                strict,
            )

        for case_name in EXPECTED_CASES:
            case_dir = domain_dir / case_name
            tsv_files = sorted(case_dir.glob("*.tsv"))
            if len(tsv_files) != 1:
                warn_or_raise(
                    f"{split_dir}/{pair_dir.name}/{domain}/{case_name}: "
                    f"expected 1 TSV, found {len(tsv_files)}",
                    strict,
                )

            row_count = 0
            for tsv_file in tsv_files:
                with tsv_file.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        row_count += 1
                        utility = float(row["my_util"])
                        utilities.append(utility)
                        if utility != 0.0:
                            agreements += 1

            if row_count != EXPECTED_ROWS_PER_CASE:
                warn_or_raise(
                    f"{split_dir}/{pair_dir.name}/{domain}/{case_name}: "
                    f"expected {EXPECTED_ROWS_PER_CASE} rows, found {row_count}",
                    strict,
                )

    if not utilities:
        return DomainStats(utility=0.0, agreement_rate=0.0, count=0)
    return DomainStats(
        utility=sum(utilities) / len(utilities),
        agreement_rate=agreements / len(utilities) * 100.0,
        count=len(utilities),
    )


def aggregate(strict: bool) -> dict[str, dict[str, DomainStats]]:
    results: dict[str, dict[str, DomainStats]] = {}
    for series_name, split_dir in SERIES:
        results[series_name] = {}
        for _, domain, _ in DOMAIN_META:
            results[series_name][domain] = collect_domain_stats(split_dir, domain, strict)
    return results


def best_series(results: dict, domain: str, metric: str) -> str:
    left = results["MiPN_general"][domain]
    right = results["Transformer_general"][domain]
    left_value = getattr(left, metric)
    right_value = getattr(right, metric)
    return "MiPN_general" if left_value >= right_value else "Transformer_general"


def format_utility(value: float, is_best: bool, latex: bool = False) -> str:
    text = f"{value:.3f}"
    if not is_best:
        return text
    return f"\\textbf{{{text}}}" if latex else f"**{text}**"


def format_rate(value: float, is_best: bool, latex: bool = False) -> str:
    text = f"{value:.1f}%"
    if latex:
        text = text.replace("%", "\\%")
        return f"\\textbf{{{text}}}" if is_best else text
    return f"**{text}**" if is_best else text


def write_csv_output(path: Path, results: dict) -> None:
    rows = [["domain", "omega", "series", "individual_utility", "agreement_rate", "count"]]
    omega_by_domain = {domain: omega for _, domain, omega in DOMAIN_META}
    for _, domain, _ in DOMAIN_META:
        for series_name, _ in SERIES:
            stats = results[series_name][domain]
            rows.append(
                [
                    domain,
                    omega_by_domain[domain],
                    series_name,
                    f"{stats.utility:.6f}",
                    f"{stats.agreement_rate:.6f}",
                    str(stats.count),
                ]
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def markdown_section(domains: tuple[tuple[str, str, str], ...], results: dict) -> list[str]:
    header = ["", ""] + [domain for _, domain, _ in domains]
    rows = [
        ["Business Area", ""] + [area for area, _, _ in domains],
        ["Domain", ""] + [domain for _, domain, _ in domains],
        ["|Omega|", ""] + [omega for _, _, omega in domains],
    ]
    for metric_label, metric, formatter in (
        ("Individual Utility", "utility", format_utility),
        ("Agreement Rate", "agreement_rate", format_rate),
    ):
        for row_index, (series_name, _) in enumerate(SERIES):
            label = metric_label if row_index == 0 else ""
            row = [label, series_name]
            for _, domain, _ in domains:
                is_best = best_series(results, domain, metric) == series_name
                value = getattr(results[series_name][domain], metric)
                row.append(formatter(value, is_best))
            rows.append(row)

    widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
    lines = []
    lines.append("| " + " | ".join(header[index].ljust(widths[index]) for index in range(len(header))) + " |")
    lines.append("| " + " | ".join("-" * widths[index] for index in range(len(header))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(row))) + " |")
    return lines


def write_markdown_output(path: Path, results: dict) -> None:
    lines = markdown_section(TOP_DOMAINS, results)
    lines.append("")
    lines.extend(markdown_section(BOTTOM_DOMAINS, results))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_row(label: str, series: str, values: list[str]) -> str:
    return " & ".join([label, series, *values]) + r" \\"


def latex_section(domains: tuple[tuple[str, str, str], ...], results: dict) -> str:
    column_spec = "ll" + "c" * len(domains)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Business Area & "
        + " & ".join(["", *[area for area, _, _ in domains]])
        + r" \\",
        "Domain & " + " & ".join(["", *[domain for _, domain, _ in domains]]) + r" \\",
        r"$|\Omega|$ & " + " & ".join(["", *[omega for _, _, omega in domains]]) + r" \\",
        r"\midrule",
    ]

    for metric_label, metric, formatter in (
        ("Individual Utility", "utility", format_utility),
        ("Agreement Rate", "agreement_rate", format_rate),
    ):
        for row_index, (series_name, _) in enumerate(SERIES):
            label = metric_label if row_index == 0 else ""
            values = []
            for _, domain, _ in domains:
                is_best = best_series(results, domain, metric) == series_name
                value = getattr(results[series_name][domain], metric)
                values.append(formatter(value, is_best, latex=True))
            lines.append(latex_row(label, series_name, values))
        if metric_label == "Individual Utility":
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_latex_output(path: Path, results: dict) -> None:
    content = "\n\n".join(
        [
            latex_section(TOP_DOMAINS, results),
            latex_section(BOTTOM_DOMAINS, results),
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def html_value(text: str) -> str:
    if text.startswith("**") and text.endswith("**"):
        return f"<strong>{html.escape(text[2:-2])}</strong>"
    return html.escape(text)


def html_section(domains: tuple[tuple[str, str, str], ...], results: dict) -> str:
    rows = [
        ["Business Area", ""] + [area for area, _, _ in domains],
        ["Domain", ""] + [domain for _, domain, _ in domains],
        ["|Omega|", ""] + [omega for _, _, omega in domains],
    ]
    for metric_label, metric, formatter in (
        ("Individual Utility", "utility", format_utility),
        ("Agreement Rate", "agreement_rate", format_rate),
    ):
        for row_index, (series_name, _) in enumerate(SERIES):
            label = metric_label if row_index == 0 else ""
            row = [label, series_name]
            for _, domain, _ in domains:
                is_best = best_series(results, domain, metric) == series_name
                value = getattr(results[series_name][domain], metric)
                row.append(formatter(value, is_best))
            rows.append(row)

    lines = ['<table class="comparison">']
    for row_index, row in enumerate(rows):
        tag = "th" if row_index < 3 else "td"
        class_name = " class=\"separator\"" if row_index in (3, 5) else ""
        lines.append(f"  <tr{class_name}>")
        for cell in row:
            lines.append(f"    <{tag}>{html_value(cell)}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def write_html_output(path: Path, results: dict) -> None:
    style = """
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>General Comparison Table</title>
<style>
body {
  color: #111;
  font-family: "Times New Roman", Times, serif;
  margin: 36px;
}
.comparison {
  border-collapse: collapse;
  font-size: 24px;
  margin: 0 auto 18px auto;
}
.comparison th,
.comparison td {
  padding: 7px 18px;
  text-align: center;
  white-space: nowrap;
}
.comparison th:first-child,
.comparison td:first-child {
  text-align: center;
}
.comparison {
  border-top: 2px solid #111;
  border-bottom: 2px solid #111;
}
.comparison tr:nth-child(3),
.comparison tr.separator {
  border-bottom: 1.5px solid #111;
}
strong {
  font-weight: 700;
}
</style>
"""
    body = "\n".join(
        [
            html_section(TOP_DOMAINS, results),
            html_section(BOTTOM_DOMAINS, results),
        ]
    )
    path.write_text(style + "<body>\n" + body + "\n</body>\n</html>\n", encoding="utf-8")


def svg_cell(
    x: float,
    y: float,
    text: str,
    size: int = 28,
    weight: str = "400",
    anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Times New Roman, Times, serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'dominant-baseline="middle" fill="#111">{html.escape(text)}</text>'
    )


def svg_rule(x1: float, x2: float, y: float, width: float = 1.5) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
        f'stroke="#111" stroke-width="{width:.1f}"/>'
    )


def svg_table_section(
    domains: tuple[tuple[str, str, str], ...],
    results: dict,
    x: float,
    y: float,
    domain_width: float,
) -> tuple[list[str], float]:
    label_width = 210.0
    series_width = 260.0
    row_height = 54.0
    widths = [label_width, series_width] + [domain_width] * len(domains)
    centers: list[float] = []
    current_x = x
    for width in widths:
        centers.append(current_x + width / 2)
        current_x += width
    right = current_x

    rows: list[list[tuple[str, str]]] = [
        [("Business Area", "400"), ("", "400"), *[(area, "400") for area, _, _ in domains]],
        [("Domain", "400"), ("", "400"), *[(domain, "400") for _, domain, _ in domains]],
        [("|Ω|", "400"), ("", "400"), *[(omega, "400") for _, _, omega in domains]],
    ]

    for metric_label, metric, formatter in (
        ("Individual Utility", "utility", format_utility),
        ("Agreement Rate", "agreement_rate", format_rate),
    ):
        for row_index, (series_name, _) in enumerate(SERIES):
            label = metric_label if row_index == 0 else ""
            row = [(label, "400"), (series_name, "400")]
            for _, domain, _ in domains:
                is_best = best_series(results, domain, metric) == series_name
                value = getattr(results[series_name][domain], metric)
                text = formatter(value, is_best).replace("**", "")
                row.append((text, "700" if is_best else "400"))
            rows.append(row)

    parts = [
        svg_rule(x, right, y, width=2.0),
        svg_rule(x, right, y + row_height, width=1.5),
        svg_rule(x, right, y + row_height * 3, width=1.5),
        svg_rule(x, right, y + row_height * 5, width=1.5),
        svg_rule(x, right, y + row_height * 7, width=2.0),
    ]

    for row_index, row in enumerate(rows):
        row_y = y + row_height * (row_index + 0.5)
        for column_index, (text, weight) in enumerate(row):
            parts.append(svg_cell(centers[column_index], row_y, text, weight=weight))

    return parts, y + row_height * 7


def write_svg_output(path: Path, results: dict) -> None:
    width = 1560
    height = 850
    top_parts, top_bottom = svg_table_section(
        TOP_DOMAINS,
        results,
        x=55,
        y=45,
        domain_width=250,
    )
    bottom_parts, _ = svg_table_section(
        BOTTOM_DOMAINS,
        results,
        x=195,
        y=top_bottom + 26,
        domain_width=250,
    )
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        *top_parts,
        *bottom_parts,
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = aggregate(args.strict)

    csv_path = args.output_dir / "general_comparison_table.csv"
    markdown_path = args.output_dir / "general_comparison_table.md"
    latex_path = args.output_dir / "general_comparison_table.tex"
    html_path = args.output_dir / "general_comparison_table.html"
    svg_path = args.output_dir / "general_comparison_table.svg"

    write_csv_output(csv_path, results)
    write_markdown_output(markdown_path, results)
    write_latex_output(latex_path, results)
    write_html_output(html_path, results)
    write_svg_output(svg_path, results)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {markdown_path}")
    print(f"Saved LaTeX: {latex_path}")
    print(f"Saved HTML: {html_path}")
    print(f"Saved SVG: {svg_path}")


if __name__ == "__main__":
    main()
