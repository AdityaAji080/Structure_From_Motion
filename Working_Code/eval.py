"""
eval.py - Evaluation metrics for Structure from Motion reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json


def compute_reprojection_error(tracks, images, K):
    errors = []
    
    for track in tracks:
        if track.point3d is None:
            continue
            
        track_errors = []
        for img_id, kp_idx in track.observations:
            img = images[img_id]
            if img.R is None or img.t is None:
                continue
                
            # Project 3D point to 2D
            X = track.point3d.reshape(3, 1)
            projected = K @ (img.R @ X + img.t)
            projected = projected[:2] / projected[2]
            
            # Get observed 2D keypoint
            kp = img.keypoints[kp_idx]
            observed = np.array([kp.pt[0], kp.pt[1]]).reshape(2, 1)
            
            # Compute error
            error = np.linalg.norm(projected - observed)
            track_errors.append(error)
            errors.append(error)
        
        # Store mean error for this track
        if track_errors:
            track.error = np.mean(track_errors)
    
    if not errors:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    
    return {
        "mean": np.mean(errors),
        "std": np.std(errors),
        "min": np.min(errors),
        "max": np.max(errors),
        "median": np.median(errors),
        "count": len(errors),
        "errors": errors
    }


def compute_point_cloud_density(tracks):
    valid_points = [t.point3d for t in tracks if t.point3d is not None]
    
    if not valid_points:
        return {"total_points": 0, "valid_points": 0}
    
    points = np.array(valid_points)
    
    # Compute bounding box and spatial extent
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    extent = max_coords - min_coords
    volume = np.prod(extent)
    
    # Compute average observation count per point
    obs_counts = [len(t.observations) for t in tracks if t.point3d is not None]
    
    return {
        "total_tracks": len(tracks),
        "valid_points": len(valid_points),
        "min_coords": min_coords.tolist(),
        "max_coords": max_coords.tolist(),
        "spatial_extent": extent.tolist(),
        "bounding_volume": volume,
        "avg_observations_per_point": np.mean(obs_counts),
        "median_observations_per_point": np.median(obs_counts)
    }


def compute_camera_trajectory_metrics(images):
    registered_images = [img for img in images if img.R is not None and img.t is not None]
    
    if len(registered_images) < 2:
        return {"registered_cameras": len(registered_images), "total_cameras": len(images)}
    
    # Extract camera centers (C = -R^T * t)
    camera_centers = []
    for img in registered_images:
        C = -img.R.T @ img.t
        camera_centers.append(C.flatten())
    
    camera_centers = np.array(camera_centers)
    
    # Compute trajectory length (sum of distances between consecutive cameras)
    trajectory_length = 0
    for i in range(len(camera_centers) - 1):
        trajectory_length += np.linalg.norm(camera_centers[i+1] - camera_centers[i])
    
    # Compute spatial extent
    min_pos = camera_centers.min(axis=0)
    max_pos = camera_centers.max(axis=0)
    extent = max_pos - min_pos
    
    return {
        "registered_cameras": len(registered_images),
        "total_cameras": len(images),
        "registration_rate": len(registered_images) / len(images),
        "trajectory_length": trajectory_length,
        "camera_positions_min": min_pos.tolist(),
        "camera_positions_max": max_pos.tolist(),
        "camera_positions_extent": extent.tolist(),
        "camera_centers": camera_centers.tolist()
    }


def assess_structural_completeness(tracks, threshold_error=10.0):
    total_tracks = len(tracks)
    high_quality_points = 0
    medium_quality_points = 0
    low_quality_points = 0
    outliers = 0
    
    for track in tracks:
        if track.point3d is None:
            continue
            
        if track.error is None:
            continue
            
        if track.error < 1.0:
            high_quality_points += 1
        elif track.error < 3.0:
            medium_quality_points += 1
        elif track.error < threshold_error:
            low_quality_points += 1
        else:
            outliers += 1
    
    return {
        "total_tracks": total_tracks,
        "high_quality_points": high_quality_points,  # < 1 px
        "medium_quality_points": medium_quality_points,  # 1-3 px
        "low_quality_points": low_quality_points,  # 3-10 px
        "outliers": outliers,  # > threshold
        "completeness_ratio": (high_quality_points + medium_quality_points) / total_tracks if total_tracks > 0 else 0
    }


def plot_reprojection_error_histogram(errors, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f} px')
    plt.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f} px')
    plt.xlabel('Reprojection Error (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Reprojection Errors', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_camera_trajectory(camera_centers, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    camera_centers = np.array(camera_centers)
    
    # Plot camera positions
    ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], 
               c='red', marker='o', s=50, label='Camera positions')
    
    # Plot trajectory path
    ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], 
            'b-', alpha=0.5, linewidth=2, label='Trajectory')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Camera Trajectory in 3D Space', fontsize=14)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_evaluation_report(tracks, images, K, output_dir="evaluation_results"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STRUCTURE FROM MOTION - EVALUATION REPORT")
    print("=" * 60)
    
    # 1. Reprojection Error
    print("\n[1] REPROJECTION ERROR ANALYSIS")
    print("-" * 60)
    reproj_stats = compute_reprojection_error(tracks, images, K)
    print(f"  Mean Error:       {reproj_stats['mean']:.3f} pixels")
    print(f"  Std Deviation:    {reproj_stats['std']:.3f} pixels")
    print(f"  Median Error:     {reproj_stats['median']:.3f} pixels")
    print(f"  Min Error:        {reproj_stats['min']:.3f} pixels")
    print(f"  Max Error:        {reproj_stats['max']:.3f} pixels")
    print(f"  Total Observations: {reproj_stats['count']}")
    
    # 2. Point Cloud Density
    print("\n[2] POINT CLOUD DENSITY")
    print("-" * 60)
    density_stats = compute_point_cloud_density(tracks)
    print(f"  Total Tracks:     {density_stats['total_tracks']}")
    print(f"  Valid 3D Points:  {density_stats['valid_points']}")
    print(f"  Spatial Extent:   {density_stats['spatial_extent']}")
    print(f"  Avg Observations/Point: {density_stats['avg_observations_per_point']:.2f}")
    
    # 3. Camera Trajectory
    print("\n[3] CAMERA TRAJECTORY CONSISTENCY")
    print("-" * 60)
    traj_stats = compute_camera_trajectory_metrics(images)
    print(f"  Registered Cameras: {traj_stats['registered_cameras']} / {traj_stats['total_cameras']}")
    print(f"  Registration Rate:  {traj_stats['registration_rate']*100:.1f}%")
    print(f"  Trajectory Length:  {traj_stats['trajectory_length']:.3f}")
    print(f"  Camera Extent:      {traj_stats['camera_positions_extent']}")
    
    # 4. Structural Completeness
    print("\n[4] STRUCTURAL COMPLETENESS")
    print("-" * 60)
    struct_stats = assess_structural_completeness(tracks)
    print(f"  High Quality (<1px):   {struct_stats['high_quality_points']}")
    print(f"  Medium Quality (1-3px): {struct_stats['medium_quality_points']}")
    print(f"  Low Quality (3-10px):   {struct_stats['low_quality_points']}")
    print(f"  Outliers (>10px):       {struct_stats['outliers']}")
    print(f"  Completeness Ratio:     {struct_stats['completeness_ratio']*100:.1f}%")
    
    print("\n" + "=" * 60)
    

    if reproj_stats['errors']:
        plot_reprojection_error_histogram(reproj_stats['errors'], 
                                          save_path=os.path.join(output_dir, "reprojection_error_histogram.png"))
    
    if traj_stats['camera_centers']:
        plot_camera_trajectory(traj_stats['camera_centers'], 
                              save_path=os.path.join(output_dir, "camera_trajectory.png"))
    
    all_metrics = {
        "reprojection_error": {k: v for k, v in reproj_stats.items() if k != 'errors'},
        "point_cloud_density": density_stats,
        "camera_trajectory": {k: v for k, v in traj_stats.items() if k != 'camera_centers'},
        "structural_completeness": struct_stats
    }
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_dir}/")
    print("  - metrics.json")
    print("  - reprojection_error_histogram.png")
    print("  - camera_trajectory.png")


if __name__ == "__main__":
    print("eval.py - SfM Evaluation Module")
    print("Import this module and call generate_evaluation_report(tracks, images, K)")
