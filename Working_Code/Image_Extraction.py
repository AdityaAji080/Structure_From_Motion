import cv2
import os

def extract_frames(
    video_path,
    output_folder="samples",
    every_n_frames=30,      # save every 30th frame
    by_time_seconds=None    # OR: save one frame every X seconds (overrides every_n_frames)
):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If user wants "every X seconds", convert to frame interval
    if by_time_seconds is not None and fps > 0:
        every_n_frames = int(round(fps * by_time_seconds))
        if every_n_frames <= 0:
            every_n_frames = 1

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Saving every {every_n_frames} frame(s)")
    print(f"Output folder: {output_folder}")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Save this frame?
        if frame_idx % every_n_frames == 0:
            filename = f"frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames to '{output_folder}'.")


if __name__ == "__main__":
    # ---- EDIT THESE ----
    video_file = "data/video/Hive Video.mp4"  # path to your video

    # Option 1: save every N-th frame
    extract_frames(video_file, output_folder="data/samples", every_n_frames=10)

    # Option 2: save one frame every X seconds (uncomment to use)
    # extract_frames(video_file, output_folder="frames_every_2s", by_time_seconds=2)
