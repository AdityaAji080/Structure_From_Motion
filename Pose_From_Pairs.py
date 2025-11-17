import numpy as np

def initialize_camera_poses_from_pairs(pairs, ref_idx, nimages):
    """
    Use per-pair geometry to initialize camera poses relative to ref_idx.

    Assumes that for each Pair:
      - pair.cams = (i, j)
      - pair.R, pair.t describe camera j relative to camera i
        with world frame at camera i (P1 = [I|0], P2 = [R|t]).
    """

    R_cams = [np.eye(3, dtype=np.float64) for _ in range(nimages)]
    t_cams = [np.zeros(3, dtype=np.float64) for _ in range(nimages)]
    has_pose = [False] * nimages

    # Reference camera fixed at [I | 0]
    has_pose[ref_idx] = True

    for pair in pairs:
        # Only consider pairs that actually have a pose
        if not hasattr(pair, "R") or pair.R is None or pair.t is None:
            continue

        i, j = pair.cams
        R = pair.R
        t = pair.t

        # Case 1: pair is (ref, j) → world is already ref
        if i == ref_idx:
            if not has_pose[j]:
                R_cams[j] = R
                t_cams[j] = t.ravel()
                has_pose[j] = True

        # Case 2: pair is (j, ref) → world is j, we need inverse transform
        elif j == ref_idx:
            if not has_pose[i]:
                # current R,t map world=j → cam_ref
                # we want cam_i (j) in ref frame → invert
                R_inv = R.T
                t_inv = -R_inv @ t
                R_cams[i] = R_inv
                t_cams[i] = t_inv.ravel()
                has_pose[i] = True

        # Else: neither camera is ref_idx → ignore for this simple star model

    return R_cams, t_cams, has_pose
