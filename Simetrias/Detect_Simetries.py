#!/usr/bin/env python3
"""CLI entrypoint for symmetry-based recognition pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from Simetrias.Utils.Pipelines import Detect_simetries


def _parse_kv_line(line: str) -> Dict[str, str]:
    pairs = [item.strip() for item in line.split(",") if item.strip()]
    return {k.strip(): v.strip() for k, v in (pair.split("=", 1) for pair in pairs)}


def _parse_bool(raw: str) -> bool:
    return raw.lower() in {"1", "true", "yes", "y"}


def _iter_processes(config_file: Path) -> Iterable[Dict[str, str]]:
    for raw_line in config_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield _parse_kv_line(line)


def _run_symmetry_process(process_args: Dict[str, str]):
    return Detect_simetries(
        path=process_args["path"],
        visualization=_parse_bool(process_args.get("visualization", "True")),
        geometry_type=process_args.get("geometry_type", "pointCloud"),
        voxel_down_sample=float(process_args.get("voxel_down_sample", 0.02)),
        NN_for_signature_build=int(process_args.get("NN_for_signature_build", 30)),
        random_frac=float(process_args.get("random_frac", 0.1 / 16)),
        filtered_SS=_parse_bool(process_args.get("filtered_SS", "False")),
        Cluster_min_samples=int(process_args.get("Cluster_min_samples", 30)),
        Cluster_xi=float(process_args.get("Cluster_xi", 0.001)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run symmetry-based recognition jobs from a key=value config file "
            "(one process per line)."
        )
    )
    parser.add_argument(
        "-i",
        "--ifile",
        required=True,
        help="Path to configuration file with process definitions.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.ifile).expanduser().resolve()
    for process in _iter_processes(config_path):
        try:
            output = _run_symmetry_process(process)
            print(output)
        except Exception as exc:  # pragma: no cover - preserves current CLI behavior
            print(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
