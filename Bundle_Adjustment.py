import numpy as np
from scipy.optimize import least_squares
import cv2


def bundle_adjust_points_for_pair(pair, X_init, pts1, pts2):
    """Two-view bundle adjustment that refines only 3D points
    while keeping the camera projection matrices fixed.

    Parameters
    ----------
    pair : object
        Must have 'extrinsics' attribute: [P1, P2] with shape (3,4) each.
    X_init : ndarray, shape (N, 3)
        Initial 3D points in the reference camera frame.
    pts1, pts2 : ndarray, shape (N, 2)
        Observed pixel coordinates in image 1 and 2.
    """
    P1, P2 = pair.extrinsics  # 3x4 each

    def project(P, X):
        # X: (N,3) -> X_h: (4,N)
        X_h = np.hstack([X, np.ones((X.shape[0], 1))]).T  # (4,N)
        proj = P @ X_h                                    # (3,N)
        proj = (proj[:2, :] / proj[2, :]).T              # (N,2)
        return proj

    def residuals(x_flat):
        X = x_flat.reshape(-1, 3)             # (N,3)
        proj1 = project(P1, X)                # (N,2)
        proj2 = project(P2, X)                # (N,2)
        r1 = (proj1 - pts1).ravel()
        r2 = (proj2 - pts2).ravel()
        return np.hstack([r1, r2])

    x0 = X_init.reshape(-1)  # initial parameter vector
    res = least_squares(residuals, x0, verbose=1)

    X_opt = res.x.reshape(-1, 3)
    return X_opt, res


def bundle_adjust_multi_view(
    K,
    camera_indices,
    point_indices,
    points_2d,
    n_cams,
    n_points,
    R_init,
    t_init,
    X_init,
    fix_first_camera=True,
):


    camera_indices = np.asarray(camera_indices, dtype=int)
    point_indices = np.asarray(point_indices, dtype=int)
    points_2d = np.asarray(points_2d, dtype=float)

    assert camera_indices.shape[0] == point_indices.shape[0] == points_2d.shape[0]

    n_obs = camera_indices.shape[0]

    # Normalize shapes for R_init and t_init
    R_init_arr = np.asarray(R_init, dtype=float)  # (n_cams, 3, 3)
    t_init_arr = np.asarray(t_init, dtype=float).reshape(n_cams, 3)

    # Parameterization:
    # - camera 0 is fixed if fix_first_camera=True
    # - each other camera i has 6 params: 3 for Rodrigues rotation, 3 for translation
    if fix_first_camera:
        cam_param_count = (n_cams - 1) * 6
    else:
        cam_param_count = n_cams * 6

    def pack_camera_params(R_arr, t_arr):
        params = []
        start_cam = 1 if fix_first_camera else 0
        for ci in range(start_cam, n_cams):
            R = R_arr[ci]
            t = t_arr[ci]
            rvec, _ = cv2.Rodrigues(R)
            params.append(np.hstack([rvec.ravel(), t.ravel()]))
        return np.hstack(params)

    # Build initial camera parameter vector
    cam_params0 = pack_camera_params(R_init_arr, t_init_arr)

    # Initial point parameters
    X0 = np.asarray(X_init, dtype=float).reshape(n_points, 3)

    x0 = np.hstack([cam_params0.ravel(), X0.ravel()])

    def unpack_camera_params(cam_params_flat):
        R_list = []
        t_list = []
        idx = 0
        start_cam = 1 if fix_first_camera else 0

        if fix_first_camera:
            # Camera 0 fixed as identity + zero translation
            R_list.append(np.eye(3))
            t_list.append(np.zeros(3))

        for ci in range(start_cam, n_cams):
            rvec = cam_params_flat[idx:idx + 3]
            t = cam_params_flat[idx + 3:idx + 6]
            idx += 6
            R, _ = cv2.Rodrigues(rvec)
            R_list.append(R)
            t_list.append(t)

        R_arr = np.stack(R_list, axis=0)
        t_arr = np.stack(t_list, axis=0)
        return R_arr, t_arr

    def project_point(K, R, t, X):
        """Project a single 3D point X into a camera with pose (R, t)."""
        X_cam = R @ X + t
        x = K @ X_cam
        u = x[0] / x[2]
        v = x[1] / x[2]
        return np.array([u, v])

    def residuals(x_flat):
        cam_params_flat = x_flat[:cam_param_count]
        pts_flat = x_flat[cam_param_count:]
        X = pts_flat.reshape(n_points, 3)

        # Build camera poses from current parameters
        R_arr, t_arr = unpack_camera_params(cam_params_flat)

        res = np.empty(2 * n_obs, dtype=float)

        for k in range(n_obs):
            ci = camera_indices[k]
            pi = point_indices[k]

            X_j = X[pi]
            R = R_arr[ci]
            t = t_arr[ci]

            u_pred, v_pred = project_point(K, R, t, X_j)
            u_obs, v_obs = points_2d[k]

            res[2 * k] = u_pred - u_obs
            res[2 * k + 1] = v_pred - v_obs

        return res

    # Run optimization
    res = least_squares(
        residuals,
        x0,
        verbose=1,
        x_scale="jac",
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        max_nfev=50,
    )

    # Unpack optimized parameters
    cam_params_opt = res.x[:cam_param_count]
    pts_opt_flat = res.x[cam_param_count:]
    X_opt = pts_opt_flat.reshape(n_points, 3)

    R_opt, t_opt = unpack_camera_params(cam_params_opt)

    return X_opt, R_opt, t_opt, res
