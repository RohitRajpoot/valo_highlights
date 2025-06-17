import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
import mlflow
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

# Compute project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Paths to features and annotations
FRAME_FEAT_DIR = os.path.join(PROJECT_ROOT, "models", "frame_feats")
AUDIO_FEAT_DIR = os.path.join(PROJECT_ROOT, "models", "audio_feats")
ANNOTATIONS_CSV = os.path.join(PROJECT_ROOT, "annotations", "with_ocr.csv")

# Hyperparameters
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 1

class HighlightDataset(Dataset):
    def __init__(self, annotations_csv):
        self.df = pd.read_csv(annotations_csv)
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {lab: i for i, lab in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row['video_name']
        base = os.path.splitext(os.path.basename(video_name))[0]
        # Load features
        frame_path = os.path.join(FRAME_FEAT_DIR, f"{base}.npy")
        audio_path = os.path.join(AUDIO_FEAT_DIR, f"{base}.npy")
        frame_feats = np.load(frame_path)
        audio_feats = np.load(audio_path)
        label = self.label2idx[row['label']]
        # Convert to tensors
        frame_tensor = torch.from_numpy(frame_feats).float()
        audio_tensor = torch.from_numpy(audio_feats).float()
        return frame_tensor, audio_tensor, label

# Collate that pads variable-length sequences
def collate_fn(batch):
    frames, audios, labels = zip(*batch)
    # Pad frames to max length in batch
    max_frame_len = max(f.shape[0] for f in frames)
    padded_frames = []
    for f in frames:
        pad_len = max_frame_len - f.shape[0]
        padded = F.pad(f, (0, 0, 0, pad_len))  # pad (W_left,W_right,H_top,H_bottom)
        padded_frames.append(padded)
    frames_tensor = torch.stack(padded_frames)

    # Pad audios to max width in batch
    max_audio_len = max(a.shape[1] for a in audios)
    padded_audios = []
    for a in audios:
        pad_width = max_audio_len - a.shape[1]
        padded = F.pad(a, (0, pad_width, 0, 0))
        padded_audios.append(padded)
    audios_tensor = torch.stack(padded_audios)

    labels_tensor = torch.tensor(labels)
    return frames_tensor, audios_tensor, labels_tensor

class HighlightModel(nn.Module):
    def __init__(self, frame_dim=512, audio_dim=64, hidden_size=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=frame_dim + audio_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, frames, audios):
        # audios: (B, n_mels, T') -> average over time dim
        audio_avg = audios.mean(dim=2)  # (B, n_mels)
        B, T, _ = frames.shape
        audio_rep = audio_avg.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([frames, audio_rep], dim=2)
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        logits = self.fc(last)
        return logits


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HighlightDataset(ANNOTATIONS_CSV)
    # Example to get audio_dim
    labels_array = dataset.df['label'].values
    classes = np.array(dataset.labels)
    weights_np = compute_class_weight("balanced", classes=classes, y=labels_array)
    class_weights = torch.tensor(weights_np, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    sample_frame, sample_audio, _ = dataset[0]
    audio_dim_example = sample_audio.shape[0]
    sample_weights = [
        class_weights[dataset.label2idx[label]].item()
        for label in dataset.df["label"].values
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_fn
    )

    model = HighlightModel(frame_dim=sample_frame.shape[1],
                           audio_dim=audio_dim_example,
                           hidden_size=HIDDEN_SIZE,
                           num_layers=NUM_LSTM_LAYERS,
                           num_classes=len(dataset.labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.start_run()
    mlflow.log_params({"lr": LR, "batch_size": BATCH_SIZE,
                       "epochs": EPOCHS, "hidden_size": HIDDEN_SIZE})

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for frames, audios, labels in dataloader:
            frames, audios, labels = frames.to(device), audios.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(frames, audios)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * frames.size(0)
        avg_loss = total_loss / len(dataset)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    mlflow.pytorch.log_model(model, "highlight_model")
    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "models", "highlight_model.pt"))
    mlflow.end_run()

if __name__ == "__main__":
    train()
