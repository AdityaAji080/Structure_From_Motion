from dataclasses import dataclass, field
import numpy as np
import cv2

@dataclass
class Pair:
    cams: tuple[int, int]                  # (index of image1, index of image2)
    matches: list[cv2.DMatch]              # list of feature matches between them
    F: np.ndarray | None = None            # 3x3 fundamental matrix
    E: np.ndarray | None = None            # 3x3 essential matrix (optional later)
    intrinsics: list[np.ndarray] = field(default_factory=list)   # K1, K2
    extrinsics: list[np.ndarray] = field(default_factory=list)   # P1, P2 (3x4)

    def update_intrinsics(self, f: float, w: float, h: float):
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float64)
        # assume same intrinsics for both cameras
        self.intrinsics = [K, K.copy()]
