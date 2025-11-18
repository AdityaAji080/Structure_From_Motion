import os
import glob
import cv2
import numpy as np
import open3d as o3d

# =====================
#  Basic Data Classes
# =====================

class ImageInfo:
    def __init__(self, img_id, name, path, image, gray):
        self.id = img_id
        self.name = name
        self.path = path
        self.image = image
        self.gray = gray

        self.keypoints = []
        self.descriptors = None

        # Pose in world frame: X_cam = R * X_world + t
        self.R = None
        self.t = None


class Track:
    def __init__(self, track_id):
        self.id = track_id
        # List of (image_id, keypoint_index)
        self.observations = []
        # 3D coordinates in world frame (np.array shape (3,))
        self.point3d = None
        self.error = None  # average reprojection error


class PairMatch:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.matches = []        # list of cv2.DMatch (raw ratio-test filtered)
        self.inlier_matches = [] # list of cv2.DMatch after geometric verification


# =====================
#  Camera / Geometry Helpers
# =====================

def estimate_intrinsics_from_image(gray):
    """Very rough intrinsics guess. Replace with real K if you know it."""
    h, w = gray.shape
    f = 1.2 * max(w, h)
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    return K


def project_point(K, R, t, X):
    """
    Project 3D point X (3x1) using camera intrinsics K and extrinsics (R, t).
    Returns 2x1 pixel coordinates.
    """
    X_cam = R @ X + t  # 3x1
    if X_cam[2, 0] <= 0:
        return None  # behind the camera
    x = K @ X_cam
    x = x / x[2, 0]
    return x[0:2, :]


def reprojection_error(K, R, t, X, x_measured):
    """
    X: 3x1, x_measured: 2x1
    Returns scalar reprojection error in pixels.
    """
    x_proj = project_point(K, R, t, X)
    if x_proj is None:
        return np.inf
    err = np.linalg.norm(x_proj - x_measured)
    return float(err)


def is_in_front_of_camera(R, t, X):
    """Check cheirality: z_cam > 0."""
    X_cam = R @ X + t
    return X_cam[2, 0] > 0


def triangulate_two_observations(img_id1, kp_idx1, img_id2, kp_idx2,
                                 images, K, reproj_error_thresh=3.0):
    """
    Triangulate a single track from two observations in two registered images.
    Returns (X (3,), avg_error) or (None, None).
    """
    img1 = images[img_id1]
    img2 = images[img_id2]
    if img1.R is None or img2.R is None:
        return None, None

    # Build projection matrices
    P1 = K @ np.hstack([img1.R, img1.t])  # 3x4
    P2 = K @ np.hstack([img2.R, img2.t])  # 3x4

    x1 = np.array(img1.keypoints[kp_idx1].pt, dtype=np.float64).reshape(2, 1)
    x2 = np.array(img2.keypoints[kp_idx2].pt, dtype=np.float64).reshape(2, 1)

    X_h = cv2.triangulatePoints(P1, P2, x1, x2)  # 4x1
    X = X_h[0:3, :] / X_h[3, 0]  # 3x1

    # Cheirality check (point must be in front of both cameras)
    if not (is_in_front_of_camera(img1.R, img1.t, X) and
            is_in_front_of_camera(img2.R, img2.t, X)):
        return None, None

    # Reprojection error in both images
    err1 = reprojection_error(K, img1.R, img1.t, X, x1)
    err2 = reprojection_error(K, img2.R, img2.t, X, x2)
    if err1 > reproj_error_thresh or err2 > reproj_error_thresh:
        return None, None

    avg_err = 0.5 * (err1 + err2)
    return X.flatten(), avg_err


# =====================
#  I/O
# =====================

def load_images(image_dir, ext="*.jpg"):
    paths = sorted(glob.glob(os.path.join(image_dir, ext)))
    images = []
    for idx, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: failed to read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        info = ImageInfo(idx, os.path.basename(path), path, img, gray)
        images.append(info)
    print(f"Loaded {len(images)} images from {image_dir}")
    return images


def save_points_to_ply(filename, tracks):
    pts = [track.point3d for track in tracks if track.point3d is not None]
    print(f"Saving {len(pts)} 3D points to {filename}")
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for X in pts:
            x, y, z = X
            f.write(f"{x} {y} {z} 255 255 255\n")


# =====================
#  Feature Extraction
# =====================

def extract_features(images):
    sift = cv2.SIFT_create(
        nfeatures=8000,
        contrastThreshold=0.02,  # lower = more keypoints in low-texture regions
        edgeThreshold=10
    )
    for img in images:
        kps, desc = sift.detectAndCompute(img.gray, None)
        img.keypoints = kps
        img.descriptors = desc
        print(f"Image {img.id} ({img.name}): {len(kps)} features")


# =====================
#  Matching + Geometric Verification
# =====================

def match_image_pairs(images, max_neighbors=10, ratio=0.8, min_matches=20):
    """
    Match each image i to neighbors j in [i+1, i+max_neighbors].
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pair_matches = {}
    n = len(images)

    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + max_neighbors)):
            desc1 = images[i].descriptors
            desc2 = images[j].descriptors
            if desc1 is None or desc2 is None:
                continue

            knn = bf.knnMatch(desc1, desc2, k=2)
            good = []
            for m_n in knn:
                if len(m_n) != 2:
                    continue
                m, n_ = m_n
                if m.distance < ratio * n_.distance:
                    good.append(m)

            if len(good) < min_matches:
                continue

            pm = PairMatch(i, j)
            pm.matches = good
            pair_matches[(i, j)] = pm
            print(f"Matched images {i}-{j}: {len(good)} good matches")

    print(f"Total candidate pairs: {len(pair_matches)}")
    return pair_matches


def geometric_verification(pair_matches, images, K,
                           ransac_thresh=1.5, min_inliers=30):
    """
    For each matched pair, estimate Essential matrix with RANSAC and keep only inlier matches.
    """
    verified = {}
    for (i, j), pm in pair_matches.items():
        kp1 = images[i].keypoints
        kp2 = images[j].keypoints

        pts1 = np.array([kp1[m.queryIdx].pt for m in pm.matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in pm.matches], dtype=np.float64)

        if len(pts1) < 8:
            continue

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh
        )
        if E is None:
            continue

        mask = mask.ravel().astype(bool)
        inliers = [m for m, inl in zip(pm.matches, mask) if inl]

        if len(inliers) < min_inliers:
            continue

        pm.inlier_matches = inliers
        verified[(i, j)] = pm
        print(f"Verified pair {i}-{j}: {len(inliers)} inliers")

    print(f"Total verified pairs: {len(verified)}")
    return verified


# =====================
#  Track Building (Union-Find)
# =====================

def build_tracks(images, verified_pairs):
    """
    Build multi-view tracks from inlier pairwise matches using union-find.
    """
    parent = {}
    rank = {}
    nodes_set = set()

    def make_set(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    # Build union-find graph
    for (i, j), pm in verified_pairs.items():
        for m in pm.inlier_matches:
            node_i = (i, m.queryIdx)
            node_j = (j, m.trainIdx)
            make_set(node_i)
            make_set(node_j)
            union(node_i, node_j)
            nodes_set.add(node_i)
            nodes_set.add(node_j)

    # Group by root
    root_to_nodes = {}
    for node in nodes_set:
        r = find(node)
        root_to_nodes.setdefault(r, []).append(node)

    tracks = []
    obs_to_track = {}
    tid = 0
    for r, nodes in root_to_nodes.items():
        if len(nodes) < 2:
            continue
        track = Track(tid)
        for (img_id, kp_idx) in nodes:
            track.observations.append((img_id, kp_idx))
            obs_to_track[(img_id, kp_idx)] = tid
        tracks.append(track)
        tid += 1

    print(f"Built {len(tracks)} tracks with >= 2 observations")
    return tracks, obs_to_track


# =====================
#  Incremental Reconstruction
# =====================

def choose_initial_pair(verified_pairs):
    """Pick pair with max inliers."""
    if not verified_pairs:
        return None
    best_key = max(verified_pairs, key=lambda k: len(verified_pairs[k].inlier_matches))
    return verified_pairs[best_key]


def initialize_two_view_reconstruction(seed_pair, images, K):
    """
    Use Essential matrix and recoverPose to initialize first two camera poses.
    """
    i0, j0 = seed_pair.i, seed_pair.j
    kp1 = images[i0].keypoints
    kp2 = images[j0].keypoints

    pts1 = np.array([kp1[m.queryIdx].pt for m in seed_pair.inlier_matches],
                    dtype=np.float64)
    pts2 = np.array([kp2[m.trainIdx].pt for m in seed_pair.inlier_matches],
                    dtype=np.float64)

    E, _ = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        print("Failed to compute E for seed pair")
        return False

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    print(f"Initial relative pose from {i0} to {j0}:\nR=\n{R}\nt=\n{t.ravel()}")

    # Fix world frame at image i0
    images[i0].R = np.eye(3, dtype=np.float64)
    images[i0].t = np.zeros((3, 1), dtype=np.float64)

    images[j0].R = R
    images[j0].t = t

    return True


def triangulate_initial_points(seed_pair, images, K, tracks,
                               reproj_error_thresh=3.0):
    i0, j0 = seed_pair.i, seed_pair.j
    count = 0
    for track in tracks:
        if track.point3d is not None:
            continue

        obs_dict = {img_id: kp_idx for (img_id, kp_idx) in track.observations}
        if i0 not in obs_dict or j0 not in obs_dict:
            continue

        kp_idx1 = obs_dict[i0]
        kp_idx2 = obs_dict[j0]
        X, err = triangulate_two_observations(
            i0, kp_idx1, j0, kp_idx2, images, K, reproj_error_thresh
        )
        if X is not None:
            track.point3d = X
            track.error = err
            count += 1

    print(f"Triangulated {count} initial 3D points from seed pair {i0}-{j0}")


def register_new_images(images, tracks, K,
                        min_pnp_points=30,
                        reproj_error_thresh=3.0):
    """
    Incrementally register images using 2D-3D correspondences and PnP.
    """
    num_images = len(images)
    registered = {i for i, img in enumerate(images) if img.R is not None}

    while True:
        best_img_id = None
        best_num_inliers = 0
        best_rvec = None
        best_tvec = None

        # Try to register each unregistered image
        for img_id in range(num_images):
            if img_id in registered:
                continue

            obj_pts = []
            img_pts = []

            # Build 2D-3D correspondences from existing tracks
            for track in tracks:
                if track.point3d is None:
                    continue
                for (iid, kp_idx) in track.observations:
                    if iid == img_id:
                        obj_pts.append(track.point3d)
                        img_pts.append(images[img_id].keypoints[kp_idx].pt)
                        break

            if len(obj_pts) < min_pnp_points:
                continue

            obj_pts = np.array(obj_pts, dtype=np.float32)  # Nx3
            img_pts = np.array(img_pts, dtype=np.float32)  # Nx2

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, K, None
            )
            if not success or inliers is None:
                continue

            num_inl = len(inliers)
            if num_inl > best_num_inliers:
                best_num_inliers = num_inl
                best_img_id = img_id
                best_rvec = rvec
                best_tvec = tvec

        if best_img_id is None:
            print("No more images can be registered.")
            break

        R, _ = cv2.Rodrigues(best_rvec)
        t = best_tvec.reshape(3, 1)
        images[best_img_id].R = R
        images[best_img_id].t = t
        registered.add(best_img_id)
        print(f"Registered image {best_img_id} ({images[best_img_id].name}) "
              f"with {best_num_inliers} PnP inliers")

        # Triangulate new points that involve this image
        triangulate_new_points_for_image(best_img_id, images, tracks, K,
                                         reproj_error_thresh)

    print(f"Total registered images: {len(registered)}")


def triangulate_new_points_for_image(new_img_id, images, tracks, K,
                                     reproj_error_thresh=3.0):
    new_img = images[new_img_id]
    if new_img.R is None:
        return

    registered_ids = {i for i, img in enumerate(images) if img.R is not None}
    count = 0

    for track in tracks:
        if track.point3d is not None:
            continue

        # Check if track seen in new image
        kp_new = None
        for (iid, kp_idx) in track.observations:
            if iid == new_img_id:
                kp_new = kp_idx
                break
        if kp_new is None:
            continue

        # Find another registered image in which this track is observed
        for (iid2, kp_idx2) in track.observations:
            if iid2 == new_img_id:
                continue
            if iid2 not in registered_ids:
                continue
            X, err = triangulate_two_observations(
                new_img_id, kp_new, iid2, kp_idx2,
                images, K, reproj_error_thresh
            )
            if X is not None:
                track.point3d = X
                track.error = err
                count += 1
                break

    print(f"Triangulated {count} new points using image {new_img_id}")

def triangulate_all_tracks(images, tracks, K, reproj_error_thresh=3.0):
    """
    After all images are registered, try to triangulate every track using
    ANY pair of registered images that see it.
    """
    registered_ids = [i for i, img in enumerate(images) if img.R is not None]
    registered_set = set(registered_ids)

    count = 0
    for track in tracks:
        if track.point3d is not None:
            continue

        obs_registered = [(iid, kp_idx) for (iid, kp_idx) in track.observations
                          if iid in registered_set]
        if len(obs_registered) < 2:
            continue

        # Try all pairs of registered observations for this track
        found = False
        for a in range(len(obs_registered)):
            for b in range(a + 1, len(obs_registered)):
                img_id1, kp1 = obs_registered[a]
                img_id2, kp2 = obs_registered[b]

                X, err = triangulate_two_observations(
                    img_id1, kp1, img_id2, kp2, images, K, reproj_error_thresh
                )
                if X is not None:
                    track.point3d = X
                    track.error = err
                    count += 1
                    found = True
                    break
            if found:
                break

    print(f"[Global Triangulation] Triangulated {count} extra tracks")


def tracks_to_open3d_pcd(tracks):
    """
    Convert our Track list into an Open3D PointCloud.
    Currently colors all points white.
    """
    pts = [t.point3d for t in tracks if t.point3d is not None]
    if len(pts) == 0:
        print("No 3D points to visualize.")
        return None

    pts = np.array(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Optional: color all points white
    colors = np.ones_like(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_tracks_with_open3d(tracks):
    pcd = tracks_to_open3d_pcd(tracks)
    if pcd is None:
        return
    o3d.visualization.draw_geometries([pcd])
