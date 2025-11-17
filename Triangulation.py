import cv2
import numpy as np

def triangulate_points_for_pair(pair, all_keypoints, inlier_mask=None):
    i1, i2 = pair.cams
    kps1 = all_keypoints[i1]
    kps2 = all_keypoints[i2]

    # Use only matches that are inliers (if mask given)
    matches = pair.matches
    if inlier_mask is not None:
        inlier_mask = inlier_mask.ravel().astype(bool)
        matches = [m for m, inl in zip(matches, inlier_mask) if inl]

    if len(matches) == 0:
        return None, None, None

    # Build 2D point arrays
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # Need shape (2, N) for cv2.triangulatePoints
    pts1_h = pts1.T  # (2, N)
    pts2_h = pts2.T  # (2, N)

    P1, P2 = pair.extrinsics  # 3x4 each

    # Triangulate: returns homogeneous coordinates (4, N)
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    # Convert to 3D Euclidean: divide by last row
    X = X_h[:3, :] / X_h[3, :]

    # X is (3, N): 3D points in the coordinate frame of the first camera
    return X.T, pts1, pts2  # shapes: (N,3), (N,2), (N,2)

def reprojection_error_for_pair(pair, X, pts1, pts2):
    """
    pair: contains extrinsics [P1, P2]
    X: (N,3) 3D points in homogeneous space (already dehomogenized)
    pts1, pts2: (N,2) observed 2D points in image 1 and 2
    """
    P1, P2 = pair.extrinsics  # 3x4

    # Convert X to homogeneous (4, N)
    X_h = np.hstack([X, np.ones((X.shape[0], 1))]).T  # (4, N)

    # Project into each camera
    proj1 = P1 @ X_h  # (3, N)
    proj2 = P2 @ X_h  # (3, N)

    # Normalize: (x/z, y/z)
    proj1 = (proj1[:2, :] / proj1[2, :]).T  # (N,2)
    proj2 = (proj2[:2, :] / proj2[2, :]).T  # (N,2)

    # Compute per-point errors
    err1 = np.linalg.norm(proj1 - pts1, axis=1)  # (N,)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)  # (N,)

    # Combine – mean reprojection error over both views
    mean_err = float(np.mean(np.concatenate([err1, err2])))
    return mean_err, err1, err2

def triangulate_points(K, R, t, pts1, pts2):
    """
    Triangulate 3D points between cam0 ([I|0]) and cam_i ([R|t]).
    pts1, pts2 are Nx2 in pixel coordinates.
    Returns X: Nx3 3D points in cam0 frame.
    """
    # Projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # cam0 at origin
    P2 = K @ np.hstack([R, t])                         # cam_i pose

    # Prepare points for cv2.triangulatePoints: (2, N)
    pts1_h = pts1.T
    pts2_h = pts2.T

    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # (4, N)
    X = (X_h[:3, :] / X_h[3, :]).T                      # (N, 3)
    return X
