from Estimate_relative_pose import estimate_relative_pose
from Triangulation import triangulate_points_for_pair, reprojection_error_for_pair
from Bundle_Adjustment import bundle_adjust_points_for_pair
import os, pickle
import numpy as np


def check_chirality(pair, X, min_frac_pos: float = 0.6):
    """
    Check that enough points are in front of both cameras.
    Returns (ok, info_dict).
    """
    # depths in camera 1 (world frame = cam1)
    z1 = X[:, 2]

    # depths in camera 2: X_cam2 = R * X_world + t
    X_world = X.T  # (3, N)
    X_cam2 = pair.R @ X_world + pair.t.reshape(3, 1)
    z2 = X_cam2[2, :]

    frac_pos1 = np.mean(z1 > 0)
    frac_pos2 = np.mean(z2 > 0)
    frac_pos_both = np.mean((z1 > 0) & (z2 > 0))
    med_z1 = np.median(z1)

    info = {
        "med_z1": med_z1,
        "frac_pos1": frac_pos1,
        "frac_pos2": frac_pos2,
        "frac_pos_both": frac_pos_both,
    }

    ok = (med_z1 > 0) and (frac_pos_both >= min_frac_pos)
    return ok, info


def process_pairs_geometry(
    pairs,
    pair_masks,
    all_keypoints,
    K,
    max_err_before_ba: float = 10.0,
    good_err_threshold: float = 3.0,
    max_err_after_ba: float = 8.0,
    cache_path: str | None = None,
    use_cache: bool = True,
    save_cache: bool = True,
):
    """
    For each pair with a valid F:
      1) estimate relative pose,
      2) triangulate points,
      3) compute reprojection error,
      4) (optionally) run bundle adjustment,
      5) keep only pairs with reasonable final error and valid depths.

    With optional caching of (R, t, X) per pair index.
    Attaches R, t, X, intrinsics, extrinsics to each accepted pair.
    Returns a list of indices of successfully processed pairs.
    """

    n_pairs = len(pairs)
    cached_results = [None] * n_pairs

    # ---------- load cache if available ----------
    if use_cache and cache_path is not None and os.path.exists(cache_path):
        print("Loading geometry cache from:", cache_path)
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        old_results = cache.get("results", [])
        if len(old_results) == n_pairs:
            cached_results = old_results
        else:
            print("  Cache size mismatch; ignoring old geometry cache.")

    successful_pairs = []

    # ---------- apply cached results where possible ----------
    for k, pair in enumerate(pairs):
        entry = cached_results[k]
        if entry is None:
            continue

        # sanity check that cams line up
        if tuple(pair.cams) != tuple(entry["cams"]):
            continue

        # restore geometry
        pair.R = entry["R"]
        pair.t = entry["t"]
        pair.X = entry["X"]

        # reconstruct intrinsics & extrinsics from K, R, t
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([pair.R, pair.t.reshape(3, 1)])
        pair.intrinsics = [K, K]
        pair.extrinsics = [P1, P2]

        # depth/chirality sanity check on cached data too
        ok_depth, info = check_chirality(pair, pair.X, min_frac_pos=0.6)
        print(
            f"[CACHE] Pair {k}: depth median={info['med_z1']:.2f}, "
            f"frac_pos1={info['frac_pos1']:.2f}, "
            f"frac_pos2={info['frac_pos2']:.2f}, "
            f"frac_pos_both={info['frac_pos_both']:.2f}"
        )
        if not ok_depth:
            print(f"[CACHE] Pair {k}: invalid depths in cache, dropping cached entry.")
            pair.R = None
            pair.t = None
            pair.X = None
            cached_results[k] = None
            continue

        successful_pairs.append(k)

    print(f"Pairs restored from geometry cache: {len(successful_pairs)}")

    # ---------- compute geometry for remaining pairs ----------
    for k, pair in enumerate(pairs):
        # skip if already restored from cache
        if k in successful_pairs:
            continue

        if getattr(pair, "F", None) is None:
            continue

        inlier_mask = pair_masks[k]
        if inlier_mask is None:
            continue

        # 1) Pose
        R, t, mask_pose = estimate_relative_pose(
            pair,
            all_keypoints,
            inlier_mask=inlier_mask,
            K=K,
        )

        # 2) Triangulate 3D structure for that pair
        X, pts1, pts2 = triangulate_points_for_pair(
            pair,
            all_keypoints,
            inlier_mask=inlier_mask,
        )
        if X is None or len(X) == 0:
            print(f"Pair {k}: no points to triangulate.")
            continue

        # 3) Reprojection error (before BA)
        mean_err, err1, err2 = reprojection_error_for_pair(pair, X, pts1, pts2)
        print(f"Pair {k}: mean reprojection error (before BA) = {mean_err:.2f} pixels")

        # Hard cutoff
        if mean_err > max_err_before_ba:
            print(f"Pair {k}: error too large ({mean_err:.2f}px). Skipping this pair.")
            continue

        # If already good, skip BA
        if mean_err < good_err_threshold:
            print(f"Pair {k}: good geometry (<{good_err_threshold} px), skipping BA.")
            X_opt = X
        else:
            # 4) Bundle adjustment for this pair
            X_opt, res = bundle_adjust_points_for_pair(pair, X, pts1, pts2)
            mean_err_after, _, _ = reprojection_error_for_pair(pair, X_opt, pts1, pts2)
            print(f"Pair {k}: mean reprojection error (after BA) = {mean_err_after:.2f} pixels")

            if mean_err_after > max_err_after_ba:
                print(
                    f"Pair {k}: still large error after BA "
                    f"({mean_err_after:.2f}px). Skipping."
                )
                continue

        # --- store pose + 3D in the pair ---
        pair.R = R
        pair.t = t
        pair.X = X_opt

        # --- depth / chirality sanity check ---
        ok_depth, info = check_chirality(pair, pair.X, min_frac_pos=0.6)
        print(
            f"Pair {k}: depth median={info['med_z1']:.2f}, "
            f"frac_pos1={info['frac_pos1']:.2f}, "
            f"frac_pos2={info['frac_pos2']:.2f}, "
            f"frac_pos_both={info['frac_pos_both']:.2f}"
        )
        if not ok_depth:
            print(f"Pair {k}: invalid depths, skipping this pair.")
            pair.R = None
            pair.t = None
            pair.X = None
            cached_results[k] = None
            continue

        # construct intrinsics & extrinsics now that we accept the pair
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([pair.R, pair.t.reshape(3, 1)])
        pair.intrinsics = [K, K]
        pair.extrinsics = [P1, P2]

        # accept + cache
        successful_pairs.append(k)
        cached_results[k] = {
            "cams": tuple(pair.cams),
            "R": pair.R,
            "t": pair.t,
            "X": pair.X,
        }

    # ---------- save cache ----------
    if save_cache and cache_path is not None:
        cache = {"results": cached_results}
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print("Saved geometry cache to:", cache_path)

    print(f"Number of successfully processed pairs: {len(successful_pairs)}")
    return successful_pairs

