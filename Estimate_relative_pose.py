import numpy as np
from Pair_Class import Pair
import cv2

def estimate_relative_pose(pair: Pair, all_keypoints, inlier_mask=None, K=None):
    """Estimate relative pose (R, t) for a given Pair using a shared intrinsic K.

    Parameters
    ----------
    pair : Pair
        Two-view pair object containing camera indices and matches.
    all_keypoints : list[list[cv2.KeyPoint]]
        Detected keypoints for each image.
    inlier_mask : np.ndarray or None
        Optional mask of inliers from F-estimation (same length as pair.matches).
    K : np.ndarray, shape (3,3)
        Camera intrinsic matrix. Must be provided so intrinsics are consistent.
    """
    if K is None:
        raise ValueError("Camera intrinsics K must be provided to estimate_relative_pose.")

    # Use the same intrinsics for both views
    K1 = K
    K2 = K
    pair.intrinsics = [K1, K2]

    F = pair.F
    if F is None:
        raise ValueError("Fundamental matrix F is not set for this pair.")

    # Essential matrix: E = K2^T * F * K1
    E = K2.T @ F @ K1
    pair.E = E

    # Build point arrays from matches (pixel coordinates)
    i1, i2 = pair.cams
    kps1 = all_keypoints[i1]
    kps2 = all_keypoints[i2]

    pts1 = np.float32([kps1[m.queryIdx].pt for m in pair.matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in pair.matches])

    # If we have a mask from F, use only F-inliers for pose estimation
    if inlier_mask is not None:
        inlier_mask = inlier_mask.ravel().astype(bool)
        pts1 = pts1[inlier_mask]
        pts2 = pts2[inlier_mask]

    # Recover pose (R, t) from the essential matrix
    num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K1)

    # Projection matrices for this pair
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, t))

    pair.extrinsics = [P1, P2]

    return R, t, mask_pose


import numpy as np
import cv2

def estimate_pose_for_pair(K, kps1, kps2, matches):
    """Estimate F, E, R, t for a single pair (cam1, cam2)."""
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # Fundamental matrix with RANSAC
    F, mask_F = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99
    )
    if F is None or mask_F is None:
        return None, None, None, None, None

    inliers = mask_F.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # Essential matrix
    E = K.T @ F @ K

    # Recover pose (R, t). Points should be normalized by K.
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)

    return F, E, R, t, inliers
