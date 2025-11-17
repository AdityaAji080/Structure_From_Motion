import cv2

def detect_sift_keypoints_and_descriptors(img, sift):
    """Detect SIFT features in a single image."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # print(f"[SIFT] keypoints: {len(keypoints)}, "
    #       f"descriptors shape: {None if descriptors is None else descriptors.shape}")
    return keypoints, descriptors