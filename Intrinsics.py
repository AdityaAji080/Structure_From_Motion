import numpy as np
import cv2

def build_intrinsics_from_image(img, fov_deg=60.0):
    """
    Minimal, reasonable intrinsics:
      - fx, fy from assumed horizontal FOV
      - cx, cy at image center
    """
    h, w = img.shape[:2]

    fov_rad = np.deg2rad(fov_deg)
    # fx from horizontal FOV
    fx = 0.5 * w / np.tan(0.5 * fov_rad)
    fy = fx  # square pixels
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)

    print(f"[K] FOV = {fov_deg} deg")
    print("[K] Intrinsics:")
    print(K)
    return K
