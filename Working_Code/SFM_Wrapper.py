import os
import glob
import cv2
import numpy as np
import open3d as o3d
from Functions import (load_images, estimate_intrinsics_from_image, extract_features,
                       match_image_pairs, geometric_verification, build_tracks, triangulate_initial_points,
                       choose_initial_pair, initialize_two_view_reconstruction, register_new_images,
                       save_points_to_ply, triangulate_all_tracks, bundle_adjust_points,load_middlebury_par)

def main():
    # ---- CONFIG ----
    IMAGE_DIR = "data/samples"   # TODO: change this
    IMAGE_EXT = "*.jpg"                # or "*.png", etc.

    # ---- 1. Load images ----
    images = load_images(IMAGE_DIR, IMAGE_EXT)
    if len(images) < 2:
        print("Need at least 2 images.")
        return

    # ---- 2. Camera intrinsics ----
    # Option A: rough guess from first image size
    K = estimate_intrinsics_from_image(images[0].gray)

    # ---- 2a. Camera intrinsics from Middlebury par file ----
    # par_path = os.path.join(IMAGE_DIR, "dino_par.txt")  # adjust exact name if needed
    # cam_params = load_middlebury_par(par_path)
    #
    # # Use K from the FIRST image (all views share the same K in this dataset)
    # first_name = images[0].name
    # if first_name not in cam_params:
    #     print(f"Image {first_name} not found in {par_path}")
    #     return
    #
    # K = cam_params[first_name]["K"]
    # print("Camera intrinsics K from par file:\n", K)

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
    seed_pair = choose_initial_pair(verified_pairs, images, K)
    print(f"Using seed pair {seed_pair.i}-{seed_pair.j} as initial two-view")

    ok = initialize_two_view_reconstruction(seed_pair, images, K)
    if not ok:
        print("Failed to initialize two-view reconstruction.")
        return

    triangulate_initial_points(seed_pair, images, K, tracks)

    # ---- 7. Incremental registration for remaining images ----
    register_new_images(images, tracks, K)

    # ---- 7.5 Global triangulation pass ----
    triangulate_all_tracks(images, tracks, K, reproj_error_thresh=10.0)

    # ---- 7.6 Bundle adjustment on 3D points ----
    # bundle_adjust_points(
    #     images,
    #     tracks,
    #     K,
    #     min_obs_per_track=2,    # require at least 2 views per point
    #     max_iterations=20,      # you can tweak this
    #     outlier_thresh=10.0     # drop points with mean error > 10 px
    # )

    # ---- 9. Also save sparse point cloud to PLY (optional) ----
    save_points_to_ply("sfm_points.ply", tracks, images)
    print("Done.")

if __name__ == "__main__":
    main()
