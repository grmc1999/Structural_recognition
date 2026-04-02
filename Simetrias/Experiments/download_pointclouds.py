#!/usr/bin/env python3
"""Download point-cloud objects for symmetry experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import open3d as o3d


def download_open3d_pointclouds(output_dir: Path) -> List[Path]:
    """Download Open3D demo pointclouds and copy them into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = o3d.data.DemoICPPointClouds()
    downloaded_paths: List[Path] = []

    for src in dataset.paths:
        src_path = Path(src)
        dst_path = output_dir / src_path.name
        if not dst_path.exists():
            pointcloud = o3d.io.read_point_cloud(str(src_path))
            o3d.io.write_point_cloud(str(dst_path), pointcloud)
        downloaded_paths.append(dst_path)

    return downloaded_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download point-cloud objects for symmetry experiments.")
    parser.add_argument(
        "--output-dir",
        default="Simetrias/Data/objects",
        help="Directory where point-cloud files will be saved.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()

    pointclouds = download_open3d_pointclouds(output_dir)
    print("Downloaded/available pointclouds:")
    for path in pointclouds:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
