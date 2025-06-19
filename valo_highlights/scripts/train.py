import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow

# Project paths
PROJECT_ROOT    = os.path.dirname(os.path.dirname(__file__))
FRAME_FEAT_DIR  = os.path.join(PROJECT_ROOT, "models", "frame_feats")
AUDIO_FEAT_DIR  = os.path.join(PROJECT_ROOT, "models", "audio_feats")
ANNOTATIONS_CSV = os.path.join(PROJECT_ROOT, "annotations", "with_ocr.csv")
MODEL_OUT_PATH  = os.path.join(PROJECT_ROOT, "models", "highlight_model_with_text.pt")

# Hyperparameters
BATCH_SIZE   = 8
LR           = 1e-4
EPOCHS       = 10
MAX_FEATURES = 100  # TF-IDF output dim

class HighlightDataset(Dataset):
    def __init__(self, annotations_csv, text_vectors):
        self.df           = pd.read_csv(annotations_csv)
        self.labels       = sorted(self.df["label"].unique())
        self.label2idx    = {lab: i for i, lab in enumerate(self.labels)}
        self.text_vectors = text_vectors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row         = self.df.iloc[idx]
        base        = os.path.splitext(os.path.basename(row["video_name"]))[0]
        frame_feats = np.load(os.path.join(FRAME_FEAT_DIR, f"{base}.npy"))
        audio_feats = np.load(os.path.join(AUDIO_FEAT_DIR, f"{base}.npy"))
        text_feats  = self.text_vectors[idx]
        label       = self.label2idx[row["label"]]
        return (
            torch.from_numpy(frame_feats).float(),
            torch.from_numpy(audio_feats).float(),
            torch.from_numpy(text_feats).float(),
            label
        )

def collate_fn(batch):
    frames_list, audios_list, texts_list, labels_list = zip(*batch)

    # 1) Frame padding
    max_f = max(f.shape[0] for f in frames_list)
    padded_f = []
    for f in frames_list:
        t = f
        pad_amt = max_f - t.size(0)
        padded = F.pad(t, (0,0, 0,pad_amt))
        padded_f.append(padded)
    frames_tensor = torch.stack(padded_f, dim=0)

    # 2) Audio padding (transpose so time is first dim)
    max_t = max(a.shape[1] for a in audios_list)
    padded_a = []
    for a in audios_list:
        t = a.T
        pad_amt = max_t - t.size(0)
        padded = F.pad(t, (0,0, 0,pad_amt))
        padded_a.append(padded)
    audios_tensor = torch.stack(padded_a, dim=0)

    # 3) Text stacking
    texts_tensor = torch.stack(list(texts_list), dim=0)

    # 4) Labels
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return frames_tensor, audios_tensor, texts_tensor, labels_tensor

class HighlightModel(nn.Module):
    def __init__(self, frame_dim, audio_dim, text_dim, num_classes):
        super().__init__()
        total_dim = frame_dim + audio_dim + text_dim
        self.fc    = nn.Linear(total_dim, num_classes)

    def forward(self, frames, audios, texts):
        f_avg = frames.mean(dim=1)
        a_avg = audios.mean(dim=1)
        x     = torch.cat([f_avg, a_avg, texts], dim=1)
        return self.fc(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF-IDF text features
    df    = pd.read_csv(ANNOTATIONS_CSV)
    texts = df["ocr_text"].fillna("").tolist()
    vect  = TfidfVectorizer(max_features=MAX_FEATURES)
    try:
        text_vecs = vect.fit_transform(texts).toarray()
    except ValueError:
        text_vecs = np.zeros((len(texts), MAX_FEATURES))

    # Dataset + sampler
    dataset    = HighlightDataset(ANNOTATIONS_CSV, text_vecs)
    labels_arr = dataset.df["label"].values
    classes    = np.array(dataset.labels)
    w_np       = compute_class_weight("balanced", classes=classes, y=labels_arr)
    class_wts  = torch.tensor(w_np, dtype=torch.float32).to(device)
    sample_wts = [class_wts[dataset.label2idx[l]].item() for l in labels_arr]
    sampler    = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)

    # Model setup
    sample_f, sample_a, sample_t, _ = dataset[0]
    model      = HighlightModel(
        frame_dim   = sample_f.shape[1],
        audio_dim   = sample_a.shape[0],
        text_dim    = sample_t.shape[0],
        num_classes = len(dataset.labels)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # MLflow logging
    mlflow.start_run()
    mlflow.log_params({
        "lr": LR, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "tfidf_feats": MAX_FEATURES
    })

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for frames, audios, texts, labels in loader:
            frames, audios, texts, labels = frames.to(device), audios.to(device), texts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(frames, audios, texts)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * frames.size(0)

        avg_loss = total_loss / len(dataset)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    mlflow.pytorch.log_model(model, "highlight_model_with_text")
    mlflow.end_run()

if __name__ == "__main__":
    train()
