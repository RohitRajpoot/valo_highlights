import cv2, os, glob

__file__ = "/Users/rohitrajpoot/Git-app/valo_highlights/scripts/extract_frames.py"
# 1. Compute project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# 2. Point to raw_videos and frames under project root
RAW_DIR = os.path.join(PROJECT_ROOT, "raw_videos")
OUT_DIR = os.path.join(PROJECT_ROOT, "frames")
FPS     = 1

print("Working directory:", os.getcwd())
print("PROJECT_ROOT =", PROJECT_ROOT)
print("Looking for videos in:", RAW_DIR)
print("Found these files:", os.listdir(RAW_DIR))


def extract_frames():
    os.makedirs(OUT_DIR, exist_ok=True)
    for video_path in glob.glob(os.path.join(RAW_DIR, "*.*")):
        if not os.path.isfile(video_path):
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = os.path.join(OUT_DIR, base)
        os.makedirs(out_subdir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        interval  = max(int(video_fps / FPS), 1)
        count = saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                filename = os.path.join(out_subdir, f"{saved:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved += 1
            count += 1

        cap.release()
        print(f"Extracted {saved} frames from {video_path} into {out_subdir}")

if __name__ == "__main__":
    extract_frames()

