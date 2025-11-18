import os
import numpy as np
import open3d as o3d

# --- Change this if needed: ---
PLY_PATH = "sfm_points.ply"
# e.g. PLY_PATH = r"C:\Users\adity\OneDrive\Desktop\Python Projects\SFM\sfm_points.ply"

print("Current working dir:", os.getcwd())
print("Trying to read:", os.path.abspath(PLY_PATH))
print("File exists?", os.path.exists(PLY_PATH))

pcd = o3d.io.read_point_cloud(PLY_PATH)
print("Open3D point cloud object:", pcd)
print("Has points?       ", pcd.has_points())
print("Num points:       ", len(pcd.points))

if len(pcd.points) == 0:
    print("No points loaded – likely wrong path or empty file.")
    exit()

# Optional: inspect bounds
pts = np.asarray(pcd.points)
print("Min coords:", pts.min(axis=0))
print("Max coords:", pts.max(axis=0))
print("Center:    ", pts.mean(axis=0))

# Color points so they stand out
pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red

bbox = pcd.get_axis_aligned_bounding_box()
bbox.color = (0, 1, 0)

# Explicit camera parameters to make sure it's not just zoomed out weirdly
center = bbox.get_center()
extent = bbox.get_extent()
radius = max(extent) * 1.5 if max(extent) > 0 else 1.0

o3d.visualization.draw_geometries(
    [pcd, bbox],
    zoom=0.7,
    front=[0.0, 0.0, -1.0],
    lookat=center,
    up=[0.0, -1.0, 0.0],
)
