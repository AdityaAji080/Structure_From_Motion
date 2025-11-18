import os
import glob
import cv2
import numpy as np
import open3d as o3d
from Functions import (load_images, estimate_intrinsics_from_image, extract_features,
                       match_image_pairs, geometric_verification, build_tracks, triangulate_initial_points,
                       choose_initial_pair, initialize_two_view_reconstruction, register_new_images,
                       save_points_to_ply, triangulate_all_tracks, visualize_tracks_with_open3d)

def main():
    # ---- CONFIG ----
    IMAGE_DIR = "data/Temple"   # TODO: change this
    IMAGE_EXT = "*.png"                 # or "*.png", etc.

    # ---- 1. Load images ----
    images = load_images(IMAGE_DIR, IMAGE_EXT)
    if len(images) < 2:
        print("Need at least 2 images.")
        return

    # ---- 2. Camera intrinsics ----
    # Option A: rough guess from first image size
    K = estimate_intrinsics_from_image(images[0].gray)

    # Option B (recommended): put your real intrinsics here
    # fx = ...
    # fy = ...
    # cx = ...
    # cy = ...
    # K = np.array([[fx, 0, cx],
    #               [0, fy, cy],
    #               [0,  0,  1]], dtype=np.float64)

    print("Camera intrinsics K:\n", K)

    # ---- 3. Features ----
    extract_features(images)

    # ---- 4. Matching + geometric verification ----
    pair_matches = match_image_pairs(images, max_neighbors=3)
    verified_pairs = geometric_verification(pair_matches, images, K)

    if not verified_pairs:
        print("No valid pairs after geometric verification.")
        return

    # ---- 5. Tracks ----
    tracks, obs_to_track = build_tracks(images, verified_pairs)
    if not tracks:
        print("No tracks built; cannot proceed.")
        return

    # ---- 6. Initialize reconstruction from best seed pair ----
    seed_pair = choose_initial_pair(verified_pairs)
    print(f"Using seed pair {seed_pair.i}-{seed_pair.j} as initial two-view")

    ok = initialize_two_view_reconstruction(seed_pair, images, K)
    if not ok:
        print("Failed to initialize two-view reconstruction.")
        return

    triangulate_initial_points(seed_pair, images, K, tracks)

    # ---- 7. Incremental registration for remaining images ----
    register_new_images(images, tracks, K)

    # ---- 7.5 Global triangulation pass (if you added it) ----
    triangulate_all_tracks(images, tracks, K, reproj_error_thresh=3.0)

    # # ---- 8. Visualize with Open3D ----
    # visualize_tracks_with_open3d(tracks)

    # ---- 9. Also save sparse point cloud to PLY (optional) ----
    save_points_to_ply("sfm_points.ply", tracks)
    print("Done.")




if __name__ == "__main__":
    main()
