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
MAX_FEATURES = 100  # number of TF-IDF features

class HighlightDataset(Dataset):
    def __init__(self, annotations_csv, text_vectors):
        self.df = pd.read_csv(annotations_csv)
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {lab: i for i, lab in enumerate(self.labels)}
        self.text_vectors = text_vectors  # numpy array [N, text_dim]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base = os.path.splitext(os.path.basename(row['video_name']))[0]
        # Load frame and audio features
        frame_feats = np.load(os.path.join(FRAME_FEAT_DIR, f"{base}.npy"))
        audio_feats = np.load(os.path.join(AUDIO_FEAT_DIR, f"{base}.npy"))
        text_feats  = self.text_vectors[idx]  # 1D array
        label = self.label2idx[row['label']]

        # Convert to tensors
        frame_tensor = torch.from_numpy(frame_feats).float()
        audio_tensor = torch.from_numpy(audio_feats).float()
        text_tensor  = torch.from_numpy(text_feats).float()
        return frame_tensor, audio_tensor, text_tensor, label

# Collate with padding for frames and audios, stack text vectors
def collate_fn(batch):
    # batch is a list of tuples: (frame_feats [T_i×F], audio_feats [L_i×A], text_vec [100], label)
    frames_list, audios_list, texts_list, labels_list = zip(*batch)

    # 1) Pad & stack frame features
    #    Find max #frames in this batch
    frame_lengths = [f.shape[0] for f in frames_list]
    max_frames    = max(frame_lengths)
    #    Pad each to [max_frames × feat_dim]
    padded_frames = []
    for f in frames_list:
        tensor_f = torch.tensor(f, dtype=torch.float32)      # [T_i, feat_dim]
        pad_amt  = max_frames - tensor_f.size(0)
        # pad (left,right, top,bottom) → here only pad rows (bottom)
        padded = F.pad(tensor_f, (0,0, 0, pad_amt))           # [max_frames, feat_dim]
        padded_frames.append(padded)
    frames_tensor = torch.stack(padded_frames, dim=0)         # [B, max_frames, feat_dim]

    # 2) Pad & stack audio features correctly
    #    audios_list[i] is a NumPy array of shape [n_mels, time_i]
    #    We want shape [time_i, n_mels] for the LSTM.
    #    First, get each clip’s time length:
    time_lengths = [a.shape[1] for a in audios_list]
    max_time = max(time_lengths)

    padded_audios = []
    for a in audios_list:
        # Transpose → [time_i, n_mels]
        tensor_a = torch.tensor(a.T, dtype=torch.float32)
        # How many time‐steps to pad?
        pad_amt = max_time - tensor_a.size(0)
        # pad=(left, right, top, bottom) in (cols, rows)
        padded = F.pad(tensor_a, (0, 0, 0, pad_amt))
        padded_audios.append(padded)

    # Now every tensor is [max_time, n_mels]; stack into [B, max_time, n_mels]
    audios_tensor = torch.stack(padded_audios, dim=0)

    # … text stacking + labels (unchanged) …

    return frames_tensor, audios_tensor, texts_tensor, labels_tensor


class HighlightModel(nn.Module):
    def __init__(self, frame_dim, audio_dim, text_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=frame_dim + audio_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size + text_dim, num_classes)

    def forward(self, frames, audios, texts):
        # Sequence fusion
        audio_avg = audios.mean(dim=2)  # (B, n_mels)
        B, T, _ = frames.shape
        audio_rep = audio_avg.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([frames, audio_rep], dim=2)
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]  # (B, hidden_size)
        # Concatenate text features
        combined = torch.cat([last, texts], dim=1)
        logits = self.fc(combined)
        return logits


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load raw annotations to build text features
    df = pd.read_csv(ANNOTATIONS_CSV)
    texts = df['ocr_text'].fillna("").values
    # TF-IDF vectorization with fallback for empty vocab
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    try:
        text_vectors = vectorizer.fit_transform(texts).toarray()  # shape [N, MAX_FEATURES]
    except ValueError:
        # No tokens found: fallback to zeros
        text_vectors = np.zeros((len(texts), MAX_FEATURES))

    # Initialize dataset and compute class weights
    dataset = HighlightDataset(ANNOTATIONS_CSV, text_vectors)
    labels_array = dataset.df['label'].values
    classes = np.array(dataset.labels)
    weights_np = compute_class_weight("balanced", classes=classes, y=labels_array)
    class_weights = torch.tensor(weights_np, dtype=torch.float).to(device)
    # Weighted sampler
    sample_weights = [class_weights[dataset.label2idx[lbl]].item() for lbl in labels_array]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                             sampler=sampler, collate_fn=collate_fn)

    # Model setup
    sample_frame, sample_audio, sample_text, _ = dataset[0]
    model = HighlightModel(
        frame_dim=sample_frame.shape[1],
        audio_dim=sample_audio.shape[0],
        text_dim=sample_text.shape[0],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=len(dataset.labels)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.start_run()
    mlflow.log_params({"lr": LR, "batch_size": BATCH_SIZE,
                       "epochs": EPOCHS, "hidden_size": HIDDEN_SIZE,
                       "tfidf_features": MAX_FEATURES})

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for frames, audios, texts, labels in dataloader:
            frames, audios, texts, labels = frames.to(device), audios.to(device), texts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(frames, audios, texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * frames.size(0)
        avg_loss = total_loss / len(dataset)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save and log model
    mlflow.pytorch.log_model(model, "highlight_model")
    model_path = os.path.join(PROJECT_ROOT, "models", "highlight_model_with_text.pt")
    torch.save(model.state_dict(), model_path)
    mlflow.end_run()

if __name__ == "__main__":
    train()