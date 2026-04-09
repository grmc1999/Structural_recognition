#!/usr/bin/env python3
"""Download closed-object meshes and export them as pointclouds for symmetry experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import open3d as o3d

# Priority order: prefer canonical closed benchmark objects when available.
# We resolve class names dynamically to keep compatibility across Open3D versions.
CANDIDATE_DATASETS = [
    ("bunny", "BunnyMesh"),
    ("dragon", "DragonMesh"),
    ("teapot", "TeapotMesh"),
    ("armadillo", "ArmadilloMesh"),
    ("knot", "KnotMesh"),
]


def _resolve_mesh_datasets() -> List[Tuple[str, str]]:
    resolved: List[Tuple[str, str]] = []
    for object_name, class_name in CANDIDATE_DATASETS:
        if hasattr(o3d.data, class_name):
            resolved.append((object_name, class_name))
    if not resolved:
        raise RuntimeError(
            "No supported closed-object mesh datasets were found in open3d.data. "
            "Please use an Open3D build that provides mesh datasets (e.g., BunnyMesh)."
        )
    return resolved


def _sample_pointcloud_from_mesh(mesh_path: Path, points_per_object: int) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to read mesh from: {mesh_path}")

    mesh.compute_vertex_normals()
    pointcloud = mesh.sample_points_poisson_disk(number_of_points=points_per_object)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    return pointcloud


def download_open3d_pointclouds(output_dir: Path, points_per_object: int = 24000) -> List[Path]:
    """Download closed meshes (bunny/dragon/teapot when available) and export pointcloud files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths: List[Path] = []
    for object_name, class_name in _resolve_mesh_datasets():
        dataset = getattr(o3d.data, class_name)()
        mesh_path = Path(dataset.path)
        dst_path = output_dir / f"{object_name}.pcd"

        if not dst_path.exists():
            pointcloud = _sample_pointcloud_from_mesh(mesh_path, points_per_object=points_per_object)
            o3d.io.write_point_cloud(str(dst_path), pointcloud)

        downloaded_paths.append(dst_path)

    return downloaded_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download closed-object pointclouds (bunny/dragon/teapot when available) for symmetry experiments."
    )
    parser.add_argument(
        "--output-dir",
        default="Simetrias/Data/objects",
        help="Directory where generated pointcloud files will be saved.",
    )
    parser.add_argument(
        "--points-per-object",
        type=int,
        default=24000,
        help="Number of points sampled from each closed mesh object.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()

    pointclouds = download_open3d_pointclouds(output_dir, points_per_object=args.points_per_object)
    print("Downloaded/available closed-object pointclouds:")
    for path in pointclouds:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
