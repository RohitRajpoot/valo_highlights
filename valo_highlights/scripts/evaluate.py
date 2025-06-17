import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from train import HighlightDataset, HighlightModel, collate_fn
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_TFIDF_FEATURES = 100

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "highlight_model_with_text.pt")
ANNO_CSV     = os.path.join(PROJECT_ROOT, "annotations", "with_ocr.csv")

def evaluate():
    # 1) Read the annotations CSV
    df = pd.read_csv(ANNO_CSV)

    # 2) Build TF–IDF vectors from the ocr_text column
    texts = df["ocr_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
    try:
        text_vectors = vectorizer.fit_transform(texts).toarray()
    except ValueError:
        # no terms found → zero‐pad
        text_vectors = np.zeros((len(texts), MAX_TFIDF_FEATURES), dtype=float)

    # 3) Now construct the dataset with both CSV and TF–IDF text features
    dataset = HighlightDataset(ANNO_CSV, text_vectors)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 2) Load model
    sample_frame, sample_audio, sample_text, _ = dataset[0]
    model = HighlightModel(
        frame_dim=sample_frame.shape[1],
        audio_dim=sample_audio.shape[0],
        text_dim=sample_text.shape[0],
        hidden_size=128,
        num_layers=1,
        num_classes=len(dataset.labels)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 3) Predict
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, audios, texts_batch, labels in loader:
            logits = model(frames, audios, texts)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # 4) Compute metrics
    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.labels)
    cm     = confusion_matrix(all_labels, all_preds)

    # Print
    print(f"Accuracy: {acc:.3f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    return acc, report, cm


if __name__ == "__main__":
    mlflow.start_run()
    acc, report, cm = evaluate()
    mlflow.log_metric("eval_accuracy", acc)
    # Save and log report
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    report_path = os.path.join(PROJECT_ROOT, "reports", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    mlflow.end_run()