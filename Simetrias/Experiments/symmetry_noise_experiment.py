#!/usr/bin/env python3
"""Run symmetry recognition on clean and noise-injected pointcloud objects."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from Simetrias.Utils.Pipelines import Detect_simetries
from Simetrias.Experiments.download_pointclouds import download_open3d_pointclouds


def _add_gaussian_noise(pointcloud: o3d.geometry.PointCloud, stddev: float, seed: int) -> o3d.geometry.PointCloud:
    noisy = o3d.geometry.PointCloud(pointcloud)
    points = np.asarray(noisy.points)
    rng = np.random.default_rng(seed)
    points = points + rng.normal(0.0, stddev, points.shape)
    noisy.points = o3d.utility.Vector3dVector(points)
    noisy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    return noisy


def _cluster_sizes(cluster_points: Sequence[np.ndarray]) -> List[int]:
    sizes: List[int] = []
    for cluster in cluster_points:
        idx = np.asarray(cluster).reshape(-1)
        sizes.append(int(np.unique(idx).shape[0]))
    return sizes


def _paint_clusters(base_cloud: o3d.geometry.PointCloud, cluster_points: Sequence[np.ndarray]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(base_cloud)
    cloud.paint_uniform_color([0.8, 0.8, 0.8])

    colors = np.array(
        [
            [0.90, 0.20, 0.20],
            [0.20, 0.70, 0.25],
            [0.15, 0.40, 0.90],
            [0.90, 0.75, 0.15],
        ]
    )

    cloud_colors = np.asarray(cloud.colors)
    for i, cluster in enumerate(cluster_points):
        indices = np.asarray(cluster).reshape(-1)
        if indices.size == 0:
            continue
        valid = indices[(indices >= 0) & (indices < len(cloud_colors))]
        cloud_colors[valid] = colors[i % len(colors)]

    return cloud


def _visualize_clean_vs_noisy(
    object_name: str,
    clean_cloud: o3d.geometry.PointCloud,
    noisy_cloud: o3d.geometry.PointCloud,
    clean_clusters: Sequence[np.ndarray],
    noisy_clusters: Sequence[np.ndarray],
):
    clean_vis = _paint_clusters(clean_cloud, clean_clusters)
    noisy_vis = _paint_clusters(noisy_cloud, noisy_clusters)

    clean_box = clean_vis.get_axis_aligned_bounding_box()
    shift = np.array([clean_box.get_extent()[0] * 1.5, 0.0, 0.0])
    noisy_vis.translate(shift)

    frame_clean = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame_noisy = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame_noisy.translate(shift)

    o3d.visualization.draw_geometries(
        [clean_vis, noisy_vis, frame_clean, frame_noisy],
        window_name=f"{object_name}: clean (left) vs noisy (right)",
    )


def _plot_cluster_size_comparison(
    object_name: str,
    clean_sizes: Sequence[int],
    noisy_sizes: Sequence[int],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    max_len = max(len(clean_sizes), len(noisy_sizes), 1)
    x = np.arange(max_len)

    clean = np.zeros(max_len, dtype=int)
    noisy = np.zeros(max_len, dtype=int)
    clean[: len(clean_sizes)] = clean_sizes
    noisy[: len(noisy_sizes)] = noisy_sizes

    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, clean, width=width, label="clean")
    plt.bar(x + width / 2, noisy, width=width, label="noisy")
    plt.xlabel("Cluster id")
    plt.ylabel("Points in cluster")
    plt.title(f"Symmetry cluster sizes: {object_name}")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{object_name}_cluster_sizes.png"
    plt.savefig(output_path, dpi=180)
    plt.close()


def _run_detection(path: Path):
    return Detect_simetries(
        path=str(path),
        visualization=False,
        geometry_type="pointCloud",
        voxel_down_sample=0.02,
        NN_for_signature_build=30,
        random_frac=0.00625,
        filtered_SS=False,
        Cluster_min_samples=30,
        Cluster_xi=0.001,
    )


def _iter_objects(paths: Iterable[Path], limit: int) -> Iterable[Path]:
    for i, path in enumerate(paths):
        if i >= limit:
            break
        yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare symmetry detection on clean vs noisy pointclouds.")
    parser.add_argument("--objects-dir", default="Simetrias/Data/objects", help="Pointcloud storage directory.")
    parser.add_argument("--artifacts-dir", default="Simetrias/Artifacts", help="Where plots and noisy clouds are stored.")
    parser.add_argument("--noise-stddev", type=float, default=0.01, help="Gaussian noise standard deviation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible noise.")
    parser.add_argument("--max-objects", type=int, default=3, help="How many downloaded objects to process.")
    parser.add_argument(
        "--skip-3d-view",
        action="store_true",
        help="Skip Open3D interactive windows and only save summary charts.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    objects_dir = Path(args.objects_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    noisy_dir = artifacts_dir / "noisy_pointclouds"
    plots_dir = artifacts_dir / "plots"
    noisy_dir.mkdir(parents=True, exist_ok=True)

    object_paths = download_open3d_pointclouds(objects_dir)

    for i, object_path in enumerate(_iter_objects(object_paths, args.max_objects)):
        object_name = object_path.stem
        clean_cloud = o3d.io.read_point_cloud(str(object_path))
        clean_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))

        noisy_cloud = _add_gaussian_noise(clean_cloud, stddev=args.noise_stddev, seed=args.seed + i)
        noisy_path = noisy_dir / f"{object_name}_noisy.pcd"
        o3d.io.write_point_cloud(str(noisy_path), noisy_cloud)

        clean_clusters = _run_detection(object_path)
        noisy_clusters = _run_detection(noisy_path)

        clean_sizes = _cluster_sizes(clean_clusters)
        noisy_sizes = _cluster_sizes(noisy_clusters)
        _plot_cluster_size_comparison(object_name, clean_sizes, noisy_sizes, plots_dir)

        if not args.skip_3d_view:
            _visualize_clean_vs_noisy(object_name, clean_cloud, noisy_cloud, clean_clusters, noisy_clusters)

        print(f"[{object_name}] clean cluster sizes: {clean_sizes}")
        print(f"[{object_name}] noisy cluster sizes: {noisy_sizes}")
        print(f"[{object_name}] noisy cloud saved to: {noisy_path}")

    print(f"Plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
