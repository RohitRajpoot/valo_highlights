import os
import json
import pandas as pd

# Compute project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Paths
OCR_DIR       = os.path.join(PROJECT_ROOT, "ocr")
ANNO_CSV      = os.path.join(PROJECT_ROOT, "annotations", "highlights.csv")
OUTPUT_CSV    = os.path.join(PROJECT_ROOT, "annotations", "with_ocr.csv")

def preprocess_ocr():
    # 1) Load the existing annotation table
    df = pd.read_csv(ANNO_CSV)

    # 2) Build a mapping from clip name → concatenated OCR text
    ocr_texts = {}
    for fname in os.listdir(OCR_DIR):
        if not fname.lower().endswith(".json"):
            continue
        clip_name = fname.replace(".json", "")
        json_path = os.path.join(OCR_DIR, fname)
        data = json.load(open(json_path, encoding="utf-8"))
        # Join all frame texts (skip empty strings)
        joined = " ".join([txt for txt in data.values() if txt.strip()])
        ocr_texts[clip_name] = joined

    # 3) Add a new column “ocr_text” by matching on video_name
    #    (Assumes your highlights.csv has a column “video_name” matching the JSON names)
    df["ocr_text"] = df["video_name"].map(ocr_texts).fillna("")

    # 4) Save to a new CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged annotations with OCR text → {OUTPUT_CSV}")

if __name__ == "__main__":
    preprocess_ocr()
