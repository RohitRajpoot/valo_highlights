#!/usr/bin/env python3
import os
import sys

# ─── Make train.py importable even when running this file as a script ───
SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow

# Import your Dataset, Model, and collate_fn from the sibling train.py
from train import HighlightDataset, HighlightModel, collate_fn

# ─── Constants ────────────────────────────────────────────────────────────
MAX_TFIDF_FEATURES = 100
PROJECT_ROOT       = os.path.dirname(SCRIPT_DIR)
MODEL_PATH         = os.path.join(PROJECT_ROOT, "models", "highlight_model_with_text.pt")
ANNO_CSV           = os.path.join(PROJECT_ROOT, "annotations", "with_ocr.csv")

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load annotations & build TF-IDF
    df    = pd.read_csv(ANNO_CSV)
    texts = df["ocr_text"].fillna("").tolist()
    vect  = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
    try:
        text_vecs = vect.fit_transform(texts).toarray()
    except ValueError:
        text_vecs = np.zeros((len(texts), MAX_TFIDF_FEATURES), dtype=float)

    # 2) Dataset & DataLoader
    dataset = HighlightDataset(ANNO_CSV, text_vecs)
    loader  = DataLoader(dataset,
                         batch_size=8,
                         shuffle=False,
                         collate_fn=collate_fn)

    # 3) Load your trained model
    sample_f, sample_a, sample_t, _ = dataset[0]
    model = HighlightModel(
        frame_dim   = sample_f.shape[1],
        audio_dim   = sample_a.shape[0],
        text_dim    = sample_t.shape[0],
        num_classes = len(dataset.labels)
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4) Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, audios, texts_batch, labels in loader:
            frames      = frames.to(device)
            audios      = audios.to(device)
            texts_batch = texts_batch.to(device)
            logits      = model(frames, audios, texts_batch)
            preds       = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # 5) Metrics
    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.labels)
    cm     = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {acc:.3f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return acc, report, cm

if __name__ == "__main__":
    mlflow.start_run()
    acc, report, cm = evaluate()
    mlflow.log_metric("eval_accuracy", acc)

    # Save the report
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    mlflow.end_run()
