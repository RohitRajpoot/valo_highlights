#!/usr/bin/env python3
"""
pipeline.py

End-to-end:
 1) cut segments
 2) extract frames
 3) extract audio
 4) extract audio features
 5) extract frame (CNN) features
 6) run OCR
 7) merge + build train table
 8) train classifier
 9) evaluate
"""
import os
import glob
import subprocess
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import pytesseract
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix

# 1) CUT SEGMENTS
def extract_segments(csv_path="annotations/Highlights.csv",
                     input_dir="raw_videos",
                     output_dir="segments_fast"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        clip = row["video_name"]            # e.g. "Clip1.webm"
        start = row["start_time"]     # e.g. "0:00:03"
        end   = row["end_time"]       # e.g. "0:00:25"
        label = row["label"]          # e.g. "ACE"
        infile  = os.path.join(input_dir, clip)
        subdir  = os.path.join(output_dir, label)
        os.makedirs(subdir, exist_ok=True)
        base    = clip.replace(".webm", f"_{start.replace(':','-')}.webm")
        outfile = os.path.join(subdir, base)
        cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-ss", start, "-i", infile, "-to", end,
            "-c:v","libvpx","-deadline","realtime","-cpu-used","5","-threads","1",
            "-c:a","libopus","-b:a","64k",
            outfile
        ]
        subprocess.run(cmd, check=True)
        print("CUT ->", outfile)

# 2) EXTRACT FRAMES
def extract_frames(input_dir="segments_fast",
                   output_dir="frames",
                   fps=1):
    os.makedirs(output_dir, exist_ok=True)
    for segdir in glob.glob(f"{input_dir}/*"):
        for webm in glob.glob(f"{segdir}/*.webm"):
            seg_id = os.path.splitext(os.path.basename(webm))[0]
            od     = os.path.join(output_dir, seg_id)
            os.makedirs(od, exist_ok=True)
            cmd = [
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", webm,
                "-vf", f"fps={fps}",
                os.path.join(od, "frame_%04d.jpg")
            ]
            subprocess.run(cmd, check=True)
    print("FRAMES extracted.")

# 3) EXTRACT AUDIO
def extract_audio(input_dir="segments_fast",
                  output_dir="audio"):
    os.makedirs(output_dir, exist_ok=True)
    for segdir in glob.glob(f"{input_dir}/*"):
        for webm in glob.glob(f"{segdir}/*.webm"):
            seg_id = os.path.splitext(os.path.basename(webm))[0]
            wav    = os.path.join(output_dir, f"{seg_id}.wav")
            cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
                   "-i", webm, wav]
            subprocess.run(cmd, check=True)
    print("AUDIO extracted.")

# 4) AUDIO FEATURES (MFCC)
def extract_audio_features(input_dir="audio",
                           output_file="features/audio_features.parquet"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    records = []
    for wav in glob.glob(f"{input_dir}/*.wav"):
        y, sr = librosa.load(wav, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccm = mfccs.mean(axis=1)
        rec = {"segment_id": os.path.splitext(os.path.basename(wav))[0]}
        rec.update({f"mfcc_{i}": mfccm[i] for i in range(len(mfccm))})
        records.append(rec)
    pd.DataFrame(records).to_parquet(output_file)
    print("Audio features ->", output_file)

# 5) FRAME FEATURES (ResNet18 embeddings)
def extract_frame_features(input_dir="frames",
                           output_file="features/frame_features.parquet",
                           model_name="resnet18"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = getattr(models, model_name)(pretrained=True).to(device).eval()
    prep   = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    records = []
    for segdir in glob.glob(f"{input_dir}/*"):
        seg_id = os.path.basename(segdir)
        feats = []
        for jpg in glob.glob(f"{segdir}/*.jpg"):
            img = Image.open(jpg).convert("RGB")
            t   = prep(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t).cpu().numpy().flatten()
            feats.append(out)
        if feats:
            avg = sum(feats)/len(feats)
            rec = {"segment_id": seg_id}
            rec.update({f"feat_{i}": avg[i] for i in range(len(avg))})
            records.append(rec)
    pd.DataFrame(records).to_parquet(output_file)
    print("Frame features ->", output_file)

# 6) OCR TEXT
def run_ocr(frames_dir="frames",
            output_file="features/ocr_text.parquet"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    records = []
    for segdir in glob.glob(f"{frames_dir}/*"):
        seg_id = os.path.basename(segdir)
        text   = ""
        for jpg in glob.glob(f"{segdir}/*.jpg"):
            text += pytesseract.image_to_string(Image.open(jpg))
        records.append({"segment_id": seg_id, "ocr_text": text})
    pd.DataFrame(records).to_parquet(output_file)
    print("OCR text ->", output_file)

# 7) MERGE FEATURES + LABELS
def prepare_data(audio_file="features/audio_features.parquet",
                 frame_file="features/frame_features.parquet",
                 ocr_file="features/ocr_text.parquet",
                 csv_path="annotations/Highlights.csv",
                 output_file="features/train_table.parquet"):
    af = pd.read_parquet(audio_file)
    ff = pd.read_parquet(frame_file)
    of = pd.read_parquet(ocr_file)
    lb = pd.read_csv(csv_path)
    # build segment_id same as step 1:
    lb["segment_id"] = (
      lb["clip"].str.replace(".webm","") + "_" +
      lb["start_time"].str.replace(":","-")
    )
    df = lb.merge(af, on="segment_id") \
           .merge(ff, on="segment_id") \
           .merge(of, on="segment_id")
    df.to_parquet(output_file)
    print("Train table ->", output_file)

# 8) TRAIN A SIMPLE MLP
def train(data_file="features/train_table.parquet",
          epochs=10, batch_size=8,
          model_out="highlight_classifier.pth"):
    df = pd.read_parquet(data_file)
    X  = df.drop(columns=["clip","start_time","end_time","label","segment_id"]).values
    y  = df["label"].values
    le = LabelEncoder().fit(y)
    y_ = le.transform(y)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = nn.Sequential(
        nn.Linear(Xt.shape[1],128),
        nn.ReLU(),
        nn.Linear(128, len(le.classes_))
    )
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epochs):
        correct = total = 0
        for xb, yb in loader:
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            preds  = logits.argmax(dim=1)
            correct += (preds==yb).sum().item()
            total   += yb.size(0)
        print(f"Epoch {e+1}/{epochs} — acc: {correct/total:.3f}")

    torch.save({"model": model.state_dict(),
                "le":    le}, model_out)
    print("Model saved to", model_out)

# 9) EVALUATE
def evaluate(model_path="highlight_classifier.pth",
             data_file="features/train_table.parquet"):
    ckpt = torch.load(model_path)
    le   = ckpt["le"]
    df   = pd.read_parquet(data_file)
    X    = df.drop(columns=["clip","start_time","end_time","label","segment_id"]).values
    y    = le.transform(df["label"].values)
    Xt   = torch.tensor(X, dtype=torch.float32)
    model= nn.Sequential(
        nn.Linear(Xt.shape[1],128),
        nn.ReLU(),
        nn.Linear(128, len(le.classes_))
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        preds = model(Xt).argmax(dim=1).numpy()
    print("Overall accuracy:", accuracy_score(y,preds))
    print("Confusion matrix:\n", confusion_matrix(y,preds))

if __name__ == "__main__":
    extract_segments()
    extract_frames()
    extract_audio()
    extract_audio_features()
    extract_frame_features()
    run_ocr()
    prepare_data()
    train()
    evaluate()
