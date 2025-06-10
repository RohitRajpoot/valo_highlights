import os
import json
import glob
from PIL import Image
import pytesseract

# Compute project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Configuration
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames")
OCR_DIR    = os.path.join(PROJECT_ROOT, "ocr")

def run_ocr():
    os.makedirs(OCR_DIR, exist_ok=True)

    # Iterate over each subfolder in frames/ (one per video)
    for clip_name in sorted(os.listdir(FRAMES_DIR)):
        clip_folder = os.path.join(FRAMES_DIR, clip_name)
        if not os.path.isdir(clip_folder):
            continue

        output_json = os.path.join(OCR_DIR, f"{clip_name}.json")
        results = {}

        # For each image in that folder
        for img_file in sorted(glob.glob(os.path.join(clip_folder, "*.jpg"))):
            img_name = os.path.basename(img_file)
            try:
                text = pytesseract.image_to_string(Image.open(img_file)).strip()
            except Exception as e:
                text = ""
            results[img_name] = text

        # Write one JSON file per video folder
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"✅ OCR completed for {clip_name} → {output_json}")

if __name__ == "__main__":
    run_ocr()
