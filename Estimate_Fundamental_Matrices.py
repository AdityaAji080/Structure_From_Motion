import os
import pickle
import numpy as np
import cv2
from Pair_Class import Pair


def estimate_F_for_pairs(
    pairs: list[Pair],
    all_keypoints: list,
    cache_path: str,
    thresh_inlier_ratio: float = 0.3,
    ransac_thresh: float = 1.0,
    confidence: float = 0.99,
):
    """Estimate fundamental matrices for all pairs, using cache if available."""

    pair_masks = [None] * len(pairs)

    # --- Try to load from cache ---
    if os.path.exists(cache_path):
        print("Loading F + masks from cache:", cache_path)
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        F_list = cache["F_list"]
        mask_list = cache["mask_list"]

        # restore into Pair objects
        for pair, F in zip(pairs, F_list):
            if F is not None:
                pair.F = F

        pair_masks = mask_list
        return F_list, mask_list, pair_masks

    # --- No cache: estimate with RANSAC ---
    print("No cache found. Estimating F with RANSAC for all pairs...")
    F_list: list = []
    mask_list: list = []

    for k, pair in enumerate(pairs):
        ind1, ind2 = pair.cams
        kps1 = all_keypoints[ind1]
        kps2 = all_keypoints[ind2]
        matches = pair.matches
        nmatches = len(matches)

        print(f"[{k + 1}/{len(pairs)}] 2-View SfM: image {ind1 + 1} vs {ind2 + 1}, matches={nmatches}")

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=confidence,
        )

        if F is None or mask is None:
            print("  Fundamental matrix estimation failed.")
            F_list.append(None)
            mask_list.append(None)
            continue

        inliers = mask.ravel().astype(bool)
        ratio_inlier = inliers.mean()
        print("  ratio of matching inliers:", ratio_inlier)

        if ratio_inlier < thresh_inlier_ratio:
            # reject this pair
            F_list.append(None)
            mask_list.append(None)
            continue

        # ensure shape (3, 3)
        if F.shape == (3, 3):
            F_use = F
        else:
            F_use = F[0].reshape(3, 3)

        pair.F = F_use
        pair_masks[k] = mask
        F_list.append(F_use)
        mask_list.append(mask)

    # --- Save cache ---
    cache = {
        "F_list": F_list,
        "mask_list": mask_list,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print("Saved F + masks cache to:", cache_path)

    return F_list, mask_list, pair_masks
