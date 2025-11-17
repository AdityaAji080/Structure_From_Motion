import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Intrinsics import build_intrinsics_from_image
from Image_reader import read_images
from SIFT import detect_sift_keypoints_and_descriptors
from Matching import build_pairs
from Estimate_Fundamental_Matrices import estimate_F_for_pairs
from Visualization_Helper import visualize_first_valid_fundamental
from Pair_Geometry import process_pairs_geometry
from Estimate_relative_pose import estimate_pose_for_pair
from Triangulation import triangulate_points
from Visualization_Helper import visualize_point_cloud
from Bundle_Adjustment import bundle_adjust_multi_view
from Pose_From_Pairs import initialize_camera_poses_from_pairs
import random

CACHE_F_FILE = "cache_F_results.pkl"
CACHE_GEOM_FILE = "cache_geometry.pkl"




image_paths = sorted(glob.glob("data/Temple/*.png"))

images, nimages = read_images(image_paths)                       #function to read images

h_img, w_img = images[0].shape[:2]                              #Calculate height and width of the image

K = build_intrinsics_from_image(images[0])                      #function to calculate K matrix

# --------------------------------------------------------------
# 1. SIFT Feature Extraction
# --------------------------------------------------------------

sift = cv2.SIFT_create(
    nfeatures = 8000,
    nOctaveLayers = 3,
    contrastThreshold = 10 / 255,                               #Impacts point cloud density and computation time
    edgeThreshold=10,
    sigma=1.6,
)

# --------------------------------------------------------------
# 2. Keypoint and Descriptor Matching
# --------------------------------------------------------------

all_keypoints = []
all_descriptors = []
image_shapes = []

for idx, img in enumerate(images):
    image_shapes.append(img.shape)
    # print(f"\n[IMG {idx}] shape: {img.shape}")
    kps, des = detect_sift_keypoints_and_descriptors(img, sift)
    all_keypoints.append(kps)
    all_descriptors.append(des)
    # print(f"[SIFT] keypoints: {len(kps)}, ")

# --------------------------------------------------------------
# 3. Pair Matching
# --------------------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

all_matches = [[None] * nimages for _ in range(nimages)]

for i in range(nimages):
    for j in range(i + 1, nimages):
        des1 = all_descriptors[i]
        des2 = all_descriptors[j]

        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        all_matches[i][j] = matches  # list of cv2.DMatch

# --------------------------------------------------------------
# 4. Pairwise matching between consecutive images
# --------------------------------------------------------------

pairs = build_pairs(all_matches, min_matches=8)

# --------------------------------------------------------------
# 5. Estimate Fundamental Matrices
# --------------------------------------------------------------

F_list, mask_list, pair_masks = estimate_F_for_pairs(
    pairs,
    all_keypoints,
    CACHE_F_FILE,
    thresh_inlier_ratio=0.3,
)

# --------------------------------------------------------------
# 6. Visualize Epipolar Geometry
# --------------------------------------------------------------

# visualize_first_valid_fundamental(pairs, pair_masks, images, all_keypoints)

# --------------------------------------------------------------
# Sanity Checks - number of good pairs, mean, min, max and median
# --------------------------------------------------------------

for k, pair in enumerate(pairs):
    if pair.F is not None:
        assert pair_masks[k] is not None
        assert len(pair_masks[k].ravel()) == len(pair.matches)

inlier_counts = []
for m in pair_masks:
    if m is None:
        continue
    inlier_counts.append(int(m.ravel().sum()))

print("Valid pairs:", len(inlier_counts))
if inlier_counts:
    print(
        "Inlier stats over valid pairs:",
        "min =", np.min(inlier_counts),
        "median =", np.median(inlier_counts),
        "max =", np.max(inlier_counts),
    )

# --------------------------------------------------------------
# 7. Pose, Triangulation, Reprojection Error, and BA
# --------------------------------------------------------------

successful_pairs = process_pairs_geometry(pairs,
                                          pair_masks,
                                          all_keypoints,
                                          K,
                                          cache_path=CACHE_GEOM_FILE,)

# --------------------------------------------------------------
# Sanity Checks - 3D pairs and some random pair checks
# --------------------------------------------------------------

all_final_depth_medians = []

for k, pair in enumerate(pairs):
    X = getattr(pair, "X", None)

    # Skip if no 3D points or empty
    if X is None or len(X) == 0:
        continue

    #skip if extrinsics are missing
    if not hasattr(pair, "extrinsics") or len(pair.extrinsics) != 2:
        continue

    z = X[:, 2]
    all_final_depth_medians.append(np.median(z))

print("Number of pairs with 3D =", len(all_final_depth_medians))

#Randomly inspect a few pairs
for _ in range(5):
    k = random.randint(0, len(pairs) - 1)
    X = getattr(pairs[k], "X", None)
    if X is not None and len(X) > 0:
        print(f"Pair {k}: X shape = {X.shape}, cams = {pairs[k].cams}")

# 3) Depth median stats across all valid pairs
all_depth_medians = []

for k, pair in enumerate(pairs):
    X = getattr(pair, "X", None)
    if X is None or len(X) == 0:
        continue

    z = X[:, 2]
    all_depth_medians.append(np.median(z))

if len(all_depth_medians) > 0:
    print(
        "Depth median stats:",
        "min =", np.min(all_depth_medians),
        "median =", np.median(all_depth_medians),
        "max =", np.max(all_depth_medians),
    )
else:
    print("No pairs with valid 3D to compute depth stats.")

# 4) Worst 10 pairs by median depth
depth_stats = []  # (k, med_z, frac_positive)

for k, pair in enumerate(pairs):
    X = getattr(pair, "X", None)
    if X is None or len(X) == 0:
        continue

    z = X[:, 2]
    med_z = np.median(z)
    frac_pos = np.mean(z > 0)
    depth_stats.append((k, med_z, frac_pos))

depth_stats.sort(key=lambda x: x[1])  # sort by median depth

print("Worst 10 pairs by median depth:")
for k, med_z, frac_pos in depth_stats[:10]:
    print(f"Pair {k}: cams={pairs[k].cams}, median z={med_z:.2f}, frac(z>0)={frac_pos:.2f}")


# --------------------------------------------------------------
# 8. Global view graph / Track building / Camera plotting
# --------------------------------------------------------------

# --- 1. Build intrinsics from image size (or reuse existing K) ---

ref_idx = 0  # reference camera index

# After you've already run process_pairs_geometry(...) and filled pair.R, pair.t
R_cams, t_cams, has_pose = initialize_camera_poses_from_pairs(
    pairs,
    ref_idx=ref_idx,
    nimages=nimages,
)

# Optional: fallback to estimate_pose_for_pair for any images
# that still don't have a pose:
for i in range(nimages):
    if i == ref_idx or has_pose[i]:
        continue

    matches = all_matches[ref_idx][i]
    if matches is None or len(matches) < 8:
        continue

    kps1 = all_keypoints[ref_idx]
    kps2 = all_keypoints[i]

    F, E, R, t, inliers = estimate_pose_for_pair(K, kps1, kps2, matches)
    if F is None:
        print(f"Fallback pose estimation failed for pair ({ref_idx}, {i})")
        continue

    R_cams[i] = R
    t_cams[i] = t.ravel()
    has_pose[i] = True

# --------------------------------------------------------------
# 9. Build feature tracks anchored on the reference image
# --------------------------------------------------------------

tracks = {}  # kp_idx in ref image -> {img_idx: kp_idx}

if nimages > 0:
    n_kp_ref = len(all_keypoints[ref_idx])
else:
    n_kp_ref = 0

for i in range(1, nimages):
    matches = all_matches[ref_idx][i]
    if matches is None:
        continue

    for m in matches:
        kp0 = m.queryIdx   # keypoint index in ref image
        kpi = m.trainIdx   # keypoint index in image i
        if kp0 >= n_kp_ref:
            continue
        if kp0 not in tracks:
            tracks[kp0] = {ref_idx: kp0}
        tracks[kp0][i] = kpi

# Keep only tracks that appear in at least 2 views (including ref)
track_list = [t for t in tracks.values() if len(t) >= 2]

print(f"Number of raw tracks (>= 2 views including ref): {len(track_list)}")

# --------------------------------------------------------------
# 10. Initialize 3D for each track via triangulation (ref & one other cam)
# --------------------------------------------------------------

X_init_list = []
valid_tracks = []

for tr in track_list:
    # Keep only observations from cameras that have a valid pose
    obs = [(img_idx, kp_idx) for img_idx, kp_idx in tr.items() if has_pose[img_idx]]
    if len(obs) < 2:
        continue

    # Ensure ref camera is present
    obs_imgs = [o[0] for o in obs]
    if ref_idx not in obs_imgs:
        continue

    # Pick reference camera and one other camera to triangulate
    other_imgs = [img for img in obs_imgs if img != ref_idx]
    if len(other_imgs) == 0:
        continue

    cam_j = other_imgs[0]

    kp_ref = tr[ref_idx]
    kp_j = tr[cam_j]

    pt_ref = np.array(all_keypoints[ref_idx][kp_ref].pt, dtype=np.float64).reshape(1, 2)
    pt_j = np.array(all_keypoints[cam_j][kp_j].pt, dtype=np.float64).reshape(1, 2)

    Rj = R_cams[cam_j]
    tj = t_cams[cam_j].reshape(3, 1)

    # Triangulate in ref frame
    X_j = triangulate_points(K, Rj, tj, pt_ref, pt_j)  # (1, 3)
    if X_j.shape[0] == 0:
        continue

    # Use this as initial 3D for the track
    X_init_list.append(X_j[0])
    valid_tracks.append(tr)

if len(valid_tracks) == 0:
    print("No valid tracks for multi-view BA. Falling back to empty point cloud.")
    points_3d = np.zeros((0, 3), dtype=np.float32)
    visualize_point_cloud(points_3d)
else:
    X_init = np.vstack(X_init_list)  # (n_points, 3)
    n_points = X_init.shape[0]
    n_cams = nimages

# --------------------------------------------------------------
# 11. Build observation arrays for BA
# --------------------------------------------------------------

    cam_indices = []
    pt_indices = []
    pts_2d = []

    for p_idx, tr in enumerate(valid_tracks):
        for img_idx, kp_idx in tr.items():
            if not has_pose[img_idx]:
                continue
            kp = all_keypoints[img_idx][kp_idx]
            u, v = kp.pt
            cam_indices.append(img_idx)
            pt_indices.append(p_idx)
            pts_2d.append([u, v])

    cam_indices = np.asarray(cam_indices, dtype=int)
    pt_indices   = np.asarray(pt_indices, dtype=int)
    pts_2d       = np.asarray(pts_2d, dtype=np.float64)

    print(f"Total observations for BA: {pts_2d.shape[0]}")
    print(f"Number of cameras: {n_cams}, number of points: {n_points}")

# --------------------------------------------------------------
# 12. Optional subsampling to keep BA fast
# --------------------------------------------------------------


    MAX_BA_POINTS = 300  # tweak this (e.g. 200–500)

    if n_points > MAX_BA_POINTS:
        keep_idx = np.random.choice(n_points, MAX_BA_POINTS, replace=False)

        # Subsample initial 3D points
        X_init = X_init[keep_idx]
        n_points = X_init.shape[0]

        # Keep only observations whose point index is in keep_idx
        keep_mask = np.isin(pt_indices, keep_idx)

        cam_indices = cam_indices[keep_mask]
        pts_2d = pts_2d[keep_mask]
        pt_indices_kept = pt_indices[keep_mask]

        # Remap old point indices to [0 .. n_points-1]
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        pt_indices = np.array([old_to_new[old] for old in pt_indices_kept], dtype=int)

        print(
            f"Subsampled BA to {n_points} points and "
            f"{pts_2d.shape[0]} observations"
        )
    else:
        print("Using all points for BA:", n_points)

# --------------------------------------------------------------
# 13. Prepare initial R, t arrays for multi-view BA
# --------------------------------------------------------------

    R_init = []
    t_init = []
    for ci in range(n_cams):
        if ci == ref_idx:
            R_init.append(np.eye(3, dtype=np.float64))
            t_init.append(np.zeros(3, dtype=np.float64))
        else:
            if has_pose[ci]:
                R_init.append(R_cams[ci])
                t_init.append(t_cams[ci])
            else:
                # Fallback: identity + zero (unlikely to be used if no observations)
                R_init.append(np.eye(3, dtype=np.float64))
                t_init.append(np.zeros(3, dtype=np.float64))

# --------------------------------------------------------------
# 14. Run multi-view bundle adjustment (or skip for debugging)
# --------------------------------------------------------------

    DEBUG_SKIP_BA = False  # set True to bypass BA and just use X_init

    if not DEBUG_SKIP_BA:
        print("Running multi-view bundle adjustment...")
        X_opt, R_opt, t_opt, res = bundle_adjust_multi_view(
            K,
            cam_indices,
            pt_indices,
            pts_2d,
            n_cams,
            n_points,
            R_init,
            t_init,
            X_init,
            fix_first_camera=True,
        )
        print("Multi-view BA finished.")
    else:
        print("Skipping multi-view BA (DEBUG_SKIP_BA=True). Using X_init.")
        X_opt = X_init
        R_opt = R_init
        t_opt = t_init

    points_3d = X_opt.astype(np.float32)

    print("points_3d (BA) shape:", points_3d.shape)

    if points_3d.shape[0] > 0:
        print("min XYZ:", points_3d.min(axis=0))
        print("max XYZ:", points_3d.max(axis=0))

    visualize_point_cloud(points_3d)

















