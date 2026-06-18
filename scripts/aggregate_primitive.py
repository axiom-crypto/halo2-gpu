#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


# Edit this list to control which primitive phases are included in the report.
# Matching is suffix-based, so "batch_invert" matches both "cuda.batch_invert"
# and "create_proof.phase1.cuda.batch_invert".
PRIMITIVES = (
    "power_of_omega",
    "generate_omega_powers",
    "generate_omega_lut",
    "fft_normal",
    "extended_from_lagrange_vec_device",
    "distribute_powers_zeta_device",
    "divide_by_vanishing_poly_device",
    "fft_normal_to_device",
    "cosetfft",
    "cosetfft_many",
    "cosetfft_many_to_device",
    "fft_many",
    "ifft_many",
    "split_radix_fft",
    "split_radix_fft_inout",
    "eval_polynomial_single",
    "basic_batch_eval_polynomial",
    "batch_eval_polynomial",
    "poly_multiply_add",
    "shplonk_rlc",
    "basic_multiopen_poly_calc",
    "multiopen_poly_calc",
    "multiexp",
    "multiexp_device_bases",
    "lookup_product",
    "permutation_product",
    "batch_invert",
    "grand_product",
    "quotient_lookups",
    "quotient_permutation",
    "permutation_quotient_gpu",
    "ifft_cosetfftpart",
    "quotient_lookups_gpu.new",
    "quotient_lookups_gpu.calculate_constraints",
    "quotient_lookups_gpu.add_permutation_constraints",
    "quotient_lookups_gpu.copy_values_back_to_host",
    "quotient_lookups_gpu.take_values_device",
    "evaluate_h.take_values_device_for_assembly",
    "evaluate_h.per_part_d2h_for_assembly",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate halo2 primitive timing metrics into markdown."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        type=Path,
        help="Path to metrics.json.",
    )
    parser.add_argument(
        "--metric_md",
        "--metric-md",
        dest="metric_md",
        type=Path,
        help="Optional markdown file to append the primitives table to.",
    )
    return parser.parse_args()


def labels_as_dict(labels) -> dict[str, str]:
    if isinstance(labels, dict):
        return {str(k): str(v) for k, v in labels.items()}
    return {str(k): str(v) for k, v in labels}


def iter_metric_records(metrics_json):
    if isinstance(metrics_json, list):
        yield from metrics_json
        return
    for kind_records in metrics_json.values():
        if isinstance(kind_records, list):
            yield from kind_records


def parse_value(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", ""))


def phase_matches_primitive(phase: str, primitive: str) -> bool:
    return phase == primitive or phase.endswith(f".{primitive}")


def format_ms(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def aggregate(metrics_json) -> dict[str, dict[str, object]]:
    rows = {primitive: {"groups": set(), "time_ms": 0.0} for primitive in PRIMITIVES}

    for record in iter_metric_records(metrics_json):
        if record.get("metric") != "halo2_section_time_ms":
            continue

        labels = labels_as_dict(record.get("labels", []))
        phase = labels.get("phase")
        group = labels.get("group")
        if not phase or not group:
            continue

        for primitive in PRIMITIVES:
            if phase_matches_primitive(phase, primitive):
                rows[primitive]["groups"].add(group)
                rows[primitive]["time_ms"] += parse_value(record.get("value", 0))
                break

    return rows


def to_markdown(rows: dict[str, dict[str, object]]) -> str:
    lines = [
        "| group | primitive | halo2_section_time_ms |",
        "| --- | --- | --- |",
    ]
    for primitive in PRIMITIVES:
        row = rows[primitive]
        groups = sorted(row["groups"])
        if not groups:
            continue
        lines.append(
            f"| {','.join(groups)} | {primitive} | {format_ms(row['time_ms'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_json = json.loads(args.json_path.read_text())
    table = to_markdown(aggregate(metrics_json))

    if args.metric_md:
        with args.metric_md.open("a") as f:
            f.write("\n\n# primitives\n\n")
            f.write(table)
            f.write("\n")
    else:
        print(table)


if __name__ == "__main__":
    main()
