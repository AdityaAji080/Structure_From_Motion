import cv2, glob

def read_images(image_paths):
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: could not load {p}")
            continue
        images.append(img)

    nimages = len(images)
    print(f"Loaded {nimages} images.")

    return images, nimages