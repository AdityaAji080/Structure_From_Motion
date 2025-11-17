import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

def _draw_epilines_on_image(img, lines, pts, title="Epipolar lines"):
    """
    img : BGR image
    lines : (N,1,3) array from cv2.computeCorrespondEpilines
    pts : (N,2) array of point coordinates corresponding to those lines
    """
    img = img.copy()
    h, w = img.shape[:2]

    for r, pt in zip(lines, pts):
        r = r[0]            # (a, b, c) in ax + by + c = 0
        a, b, c = r
        if abs(b) < 1e-6:   # avoid division by zero
            continue

        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_epipolar_geometry(img1, img2, kps1, kps2, matches, F, mask=None, max_points=20):
    """
    Visualize epipolar lines induced by F.

    img1, img2 : BGR images
    kps1, kps2 : lists of cv2.KeyPoint
    matches    : list of cv2.DMatch
    F          : 3x3 fundamental matrix
    mask       : optional (N,1) inlier mask from findFundamentalMat
    """
    # use only inliers if mask is given
    if mask is not None:
        mask = mask.ravel().astype(bool)
        matches = [m for m, inl in zip(matches, mask) if inl]

    matches = matches[:max_points]

    if len(matches) == 0:
        print("No matches to visualize.")
        return

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Epilines in image 2 for points in image 1
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)   # 1 -> points from first image
    # Epilines in image 1 for points in image 2
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)   # 2 -> points from second image

    _draw_epilines_on_image(
        img2, lines2, pts2.reshape(-1, 2),
        title="Epilines in image 2 (for points from image 1)"
    )
    _draw_epilines_on_image(
        img1, lines1, pts1.reshape(-1, 2),
        title="Epilines in image 1 (for points from image 2)"
    )


def visualize_first_valid_fundamental(
    pairs,
    pair_masks,
    images,
    all_keypoints,
):
    """Visualize epipolar geometry for the first pair with a valid F/mask."""
    valid_indices = [k for k, m in enumerate(pair_masks) if m is not None]

    if not valid_indices:
        print("No valid fundamental matrices to visualize.")
        return

    k = valid_indices[0]
    pair = pairs[k]
    mask = pair_masks[k]

    i1, i2 = pair.cams
    img1 = images[i1]
    img2 = images[i2]
    kps1 = all_keypoints[i1]
    kps2 = all_keypoints[i2]

    print(f"Visualizing epipolar geometry for pair index {k}: images {i1} & {i2}")
    visualize_epipolar_geometry(img1, img2, kps1, kps2, pair.matches, pair.F, mask)

def visualize_point_cloud(points_3d, colors=None, title="3D Reconstruction"):
    if points_3d is None or points_3d.shape[0] == 0:
        print("No 3D points to visualize.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))

    if colors is not None and colors.shape[0] == points_3d.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Center the cloud
    center = points_3d.mean(axis=0)
    pcd.translate(-center)

    # Normalize scale so it fits nicely in view
    max_norm = np.max(np.linalg.norm(points_3d - center, axis=1))
    if max_norm > 0:
        pcd.scale(1.0 / max_norm, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 3.0  # bigger dots

    vis.run()
    vis.destroy_window()