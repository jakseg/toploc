#!/usr/bin/env python3
"""
collect_toploc_results.py
=========================
Collect TopLoc IVF/IVF+/HNSW sweep CSVs into one coherent results folder.

Input CSVs supported
--------------------
1) combine_all3.py output for IVF/IVF+/baseline IVF, e.g.
   results/raw/ivf/ivf_snowflake_H1024_A0.1.csv
   or results3_snowflake_ivf.csv

2) combine_base_top_hnsw.py output for HNSW/TopLoc-HNSW, e.g.
   results/raw/hnsw/hnsw_snowflake_up2_ep1_mmap.csv
   or results_snowflake_hnsw_toploc_combined.csv

Outputs
-------
  results/combined/toploc_all_results_wide.csv
      one row per method/configuration, with metric columns.

  results/combined/toploc_all_results_long.csv
      one row per metric/configuration, easier for pivot tables and plotting.

Usage
-----
  python collect_toploc_results.py
  python collect_toploc_results.py --input-glob "results/raw/**/*.csv"
  python collect_toploc_results.py --out-dir results/combined
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_INPUT_GLOBS = [
    "results/raw/**/*.csv",
    "results3_*_ivf.csv",
    "results_*_hnsw_toploc_combined.csv",
    "results/hnsw_sweep_*.csv",
]

COMMON_FIELDS = [
    "paper",
    "dataset",
    "model",
    "index_type",
    "method",
    "variant",
    "nprobe",
    "h_cached",
    "alpha",
    "ef_search",
    "up",
    "entry_points",
    "threads",
    "mmap",
    "backend",
    "ndcg_at_3",
    "ndcg_at_10",
    "mrr_at_10",
    "followup_ms_per_query",
    "overall_ms_per_query",
    "speedup_followup",
    "refreshes",
    "avg_visited_nodes",
    "source_file",
]

METRIC_FIELDS = [
    ("NDCG@3", "ndcg_at_3", "score"),
    ("NDCG@10", "ndcg_at_10", "score"),
    ("MRR@10", "mrr_at_10", "score"),
    ("followup_ms_per_query", "followup_ms_per_query", "ms"),
    ("overall_ms_per_query", "overall_ms_per_query", "ms"),
    ("speedup_followup", "speedup_followup", "x"),
]


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_float(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    try:
        x = float(s)
    except ValueError:
        return s
    if math.isnan(x) or math.isinf(x):
        return ""
    return str(x)


def pick_model(path: str) -> str:
    m = re.search(r"(?:^|[_/.-])(snowflake|dragon)(?:[_/.-]|$)", path, re.I)
    return m.group(1).lower() if m else ""


def pick_h(path: str) -> str:
    m = re.search(r"(?:^|[_-])H(?P<h>\d+)(?:[_-]|\.|$)", path)
    return m.group("h") if m else ""


def pick_alpha(path: str) -> str:
    m = re.search(r"(?:^|[_-])A(?P<a>\d+(?:\.\d+)?)(?:[_-]|\.|$)", path)
    return m.group("a") if m else ""


def pick_up(path: str) -> str:
    m = re.search(r"(?:^|[_-])up(?P<up>\d+)(?:[_-]|\.|$)", path, re.I)
    return m.group("up") if m else ""


def pick_ep(path: str) -> str:
    m = re.search(r"(?:^|[_-])ep(?P<ep>\d+)(?:[_-]|\.|$)", path, re.I)
    return m.group("ep") if m else ""


def pick_mmap(path: str) -> str:
    return "1" if "mmap" in path.lower() else ""


def base_row(path: str) -> Dict[str, str]:
    return {
        "paper": "TopLoc",
        "dataset": "TREC_CAsT_2019",
        "model": pick_model(path),
        "index_type": "",
        "method": "",
        "variant": "",
        "nprobe": "",
        "h_cached": pick_h(path),
        "alpha": pick_alpha(path),
        "ef_search": "",
        "up": pick_up(path),
        "entry_points": pick_ep(path),
        "threads": "",
        "mmap": pick_mmap(path),
        "backend": "",
        "ndcg_at_3": "",
        "ndcg_at_10": "",
        "mrr_at_10": "",
        "followup_ms_per_query": "",
        "overall_ms_per_query": "",
        "speedup_followup": "",
        "refreshes": "",
        "avg_visited_nodes": "",
        "source_file": path,
    }


def detect_kind(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "unknown"
    cols = set(rows[0].keys())
    if "nprobe" in cols and {"baseline_fu_ms", "toploc_fu_ms", "toplocplus_fu_ms"} <= cols:
        return "ivf_all3"
    if "ef_search" in cols and {"baseline_followup_ms_per_query", "toploc_followup_ms_per_query"} <= cols:
        return "hnsw_combined"
    return "unknown"


def convert_ivf_all3(path: str, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        nprobe = r.get("nprobe", "")
        specs = [
            {
                "method": "IVF",
                "variant": "baseline",
                "fu": r.get("baseline_fu_ms"),
                "ov": r.get("baseline_overall_ms"),
                "spd": "1.0",
                "n3": r.get("baseline_NDCG@3"),
                "n10": r.get("baseline_NDCG@10"),
                "mrr": r.get("baseline_MRR@10"),
                "refreshes": "",
            },
            {
                "method": "TopLoc-IVF",
                "variant": "toploc",
                "fu": r.get("toploc_fu_ms"),
                "ov": r.get("toploc_overall_ms"),
                "spd": r.get("speedup_toploc"),
                "n3": r.get("toploc_NDCG@3"),
                "n10": r.get("toploc_NDCG@10"),
                "mrr": r.get("toploc_MRR@10"),
                "refreshes": "",
            },
            {
                "method": "TopLoc-IVF+",
                "variant": "toploc_plus",
                "fu": r.get("toplocplus_fu_ms"),
                "ov": r.get("toplocplus_overall_ms"),
                "spd": r.get("speedup_toplocplus"),
                "n3": r.get("toplocplus_NDCG@3"),
                "n10": r.get("toplocplus_NDCG@10"),
                "mrr": r.get("toplocplus_MRR@10"),
                "refreshes": r.get("toplocplus_refreshes"),
            },
        ]
        for spec in specs:
            row = base_row(path)
            row.update(
                {
                    "model": row["model"] or pick_model(path),
                    "index_type": "ivf",
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "nprobe": clean_float(nprobe),
                    "ndcg_at_3": clean_float(spec["n3"]),
                    "ndcg_at_10": clean_float(spec["n10"]),
                    "mrr_at_10": clean_float(spec["mrr"]),
                    "followup_ms_per_query": clean_float(spec["fu"]),
                    "overall_ms_per_query": clean_float(spec["ov"]),
                    "speedup_followup": clean_float(spec["spd"]),
                    "refreshes": clean_float(spec["refreshes"]),
                }
            )
            out.append(row)
    return out


def convert_hnsw_combined(path: str, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        ef = r.get("ef_search", "")
        up = r.get("up", "") or pick_up(path)
        ep = r.get("entry_points", "") or pick_ep(path)
        specs = [
            {
                "method": "HNSW",
                "variant": "baseline",
                "fu": r.get("baseline_followup_ms_per_query"),
                "ov": r.get("baseline_overall_ms_per_query"),
                "spd": "1.0",
                "n3": r.get("baseline_NDCG@3"),
                "n10": r.get("baseline_NDCG@10"),
                "mrr": r.get("baseline_MRR@10"),
                "avg_visited": "",
            },
            {
                "method": "TopLoc-HNSW",
                "variant": "toploc",
                "fu": r.get("toploc_followup_ms_per_query"),
                "ov": r.get("toploc_overall_ms_per_query"),
                "spd": r.get("speedup_followup"),
                "n3": r.get("toploc_NDCG@3"),
                "n10": r.get("toploc_NDCG@10"),
                "mrr": r.get("toploc_MRR@10"),
                "avg_visited": r.get("avg_visited_nodes_toploc"),
            },
        ]
        for spec in specs:
            row = base_row(path)
            row.update(
                {
                    "model": row["model"] or pick_model(path),
                    "index_type": "hnsw",
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "ef_search": clean_float(ef),
                    "up": clean_float(up),
                    "entry_points": clean_float(ep),
                    "threads": clean_float(r.get("threads")),
                    "backend": r.get("backend", ""),
                    "ndcg_at_3": clean_float(spec["n3"]),
                    "ndcg_at_10": clean_float(spec["n10"]),
                    "mrr_at_10": clean_float(spec["mrr"]),
                    "followup_ms_per_query": clean_float(spec["fu"]),
                    "overall_ms_per_query": clean_float(spec["ov"]),
                    "speedup_followup": clean_float(spec["spd"]),
                    "avg_visited_nodes": clean_float(spec["avg_visited"]),
                }
            )
            out.append(row)
    return out


def to_long_rows(wide_rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    long_rows: List[Dict[str, str]] = []
    id_fields = [f for f in COMMON_FIELDS if f not in {mf[1] for mf in METRIC_FIELDS}]
    for row in wide_rows:
        for metric_name, field_name, unit in METRIC_FIELDS:
            value = row.get(field_name, "")
            if value == "":
                continue
            long_row = {k: row.get(k, "") for k in id_fields}
            long_row.update({"metric": metric_name, "value": value, "unit": unit})
            long_rows.append(long_row)
    return long_rows


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expand_input_globs(patterns: List[str]) -> List[str]:
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    # unique, stable order
    return sorted(dict.fromkeys(paths))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-glob",
        action="append",
        default=None,
        help="CSV glob to read. Can be repeated. Default reads results/raw/**/*.csv and legacy result CSVs.",
    )
    ap.add_argument("--out-dir", default="results/combined")
    args = ap.parse_args()

    input_globs = args.input_glob if args.input_glob else DEFAULT_INPUT_GLOBS
    paths = expand_input_globs(input_globs)

    wide_rows: List[Dict[str, str]] = []
    skipped: List[Tuple[str, str]] = []

    for path in paths:
        try:
            rows = read_csv(path)
        except Exception as exc:
            skipped.append((path, f"read error: {exc}"))
            continue

        kind = detect_kind(rows)
        if kind == "ivf_all3":
            wide_rows.extend(convert_ivf_all3(path, rows))
        elif kind == "hnsw_combined":
            wide_rows.extend(convert_hnsw_combined(path, rows))
        else:
            skipped.append((path, "unknown CSV schema"))

    out_dir = Path(args.out_dir)
    wide_path = out_dir / "toploc_all_results_wide.csv"
    long_path = out_dir / "toploc_all_results_long.csv"

    write_csv(wide_path, wide_rows, COMMON_FIELDS)

    long_rows = to_long_rows(wide_rows)
    long_fields = [f for f in COMMON_FIELDS if f not in {mf[1] for mf in METRIC_FIELDS}] + [
        "metric",
        "value",
        "unit",
    ]
    write_csv(long_path, long_rows, long_fields)

    print(f"Read CSV files: {len(paths)}")
    print(f"Unified wide rows: {len(wide_rows)} -> {wide_path}")
    print(f"Unified long rows: {len(long_rows)} -> {long_path}")
    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"  - {path}: {reason}")


if __name__ == "__main__":
    main()
