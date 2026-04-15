#!/usr/bin/env python3
"""Create and/or download pointcloud objects for symmetry experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

# Optional benchmark meshes (depends on Open3D build/data availability)
CANDIDATE_DATASETS = [
    ("bunny", "BunnyMesh"),
    ("dragon", "DragonMesh"),
    ("teapot", "TeapotMesh"),
    ("armadillo", "ArmadilloMesh"),
    ("knot", "KnotMesh"),
]


def _sample_mesh(mesh: o3d.geometry.TriangleMesh, points_per_object: int) -> o3d.geometry.PointCloud:
    if mesh.is_empty():
        raise RuntimeError("Cannot sample an empty mesh.")
    mesh.compute_vertex_normals()
    pointcloud = mesh.sample_points_poisson_disk(number_of_points=points_per_object)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    return pointcloud


def _make_plane_pointcloud(points_per_axis: int = 160) -> o3d.geometry.PointCloud:
    """Generate a unit plane on z=0 in [0,1]x[0,1]."""
    x = np.linspace(0.0, 1.0, points_per_axis)
    y = np.linspace(0.0, 1.0, points_per_axis)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    pts = np.column_stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 1.0]]), (pts.shape[0], 1)))
    return pcd


def _make_primitive_meshes() -> Dict[str, o3d.geometry.TriangleMesh]:
    """Create simple analytic meshes centered near the origin."""
    primitives: Dict[str, o3d.geometry.TriangleMesh] = {
        "unit_cube": o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0),
        "unit_sphere": o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=40),
        "unit_cylinder": o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=1.0, resolution=60),
        "unit_cone": o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=1.0, resolution=60),
        "unit_torus": o3d.geometry.TriangleMesh.create_torus(torus_radius=0.6, tube_radius=0.2),
        "unit_octahedron": o3d.geometry.TriangleMesh.create_octahedron(radius=0.7),
        "unit_icosahedron": o3d.geometry.TriangleMesh.create_icosahedron(radius=0.7),
    }

    for mesh in primitives.values():
        mesh.translate(-mesh.get_center())

    return primitives


def _resolve_mesh_datasets() -> List[Tuple[str, str]]:
    resolved: List[Tuple[str, str]] = []
    for object_name, class_name in CANDIDATE_DATASETS:
        if hasattr(o3d.data, class_name):
            resolved.append((object_name, class_name))
    return resolved


def download_open3d_pointclouds(
    output_dir: Path,
    points_per_object: int = 24000,
    include_benchmark_meshes: bool = False,
) -> List[Path]:
    """Generate simple-shape pointclouds, with optional canonical mesh conversions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []

    # Always generate simple analytic shapes (no network dependency).
    primitive_meshes = _make_primitive_meshes()
    for name, mesh in primitive_meshes.items():
        dst_path = output_dir / f"{name}.pcd"
        if not dst_path.exists():
            pointcloud = _sample_mesh(mesh, points_per_object=points_per_object)
            o3d.io.write_point_cloud(str(dst_path), pointcloud)
        written_paths.append(dst_path)

    plane_path = output_dir / "unit_plane.pcd"
    if not plane_path.exists():
        plane_cloud = _make_plane_pointcloud()
        o3d.io.write_point_cloud(str(plane_path), plane_cloud)
    written_paths.append(plane_path)

    # Optional: add canonical closed-object meshes available in local Open3D data modules.
    if include_benchmark_meshes:
        for object_name, class_name in _resolve_mesh_datasets():
            dataset = getattr(o3d.data, class_name)()
            mesh = o3d.io.read_triangle_mesh(dataset.path)
            mesh.translate(-mesh.get_center())

            dst_path = output_dir / f"{object_name}.pcd"
            if not dst_path.exists():
                pointcloud = _sample_mesh(mesh, points_per_object=points_per_object)
                o3d.io.write_point_cloud(str(dst_path), pointcloud)
            written_paths.append(dst_path)

    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create pointclouds for symmetry experiments from simple shapes (and optional benchmark meshes)."
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
        help="Number of sampled points for mesh-based objects.",
    )
    parser.add_argument(
        "--include-benchmark-meshes",
        action="store_true",
        help="Also export bunny/dragon/teapot-like datasets when available in this Open3D build.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()

    pointclouds = download_open3d_pointclouds(
        output_dir,
        points_per_object=args.points_per_object,
        include_benchmark_meshes=args.include_benchmark_meshes,
    )
    print("Generated/available pointclouds:")
    for path in pointclouds:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
