# streamlit_app.py
import streamlit as st
import os, subprocess, tempfile, shutil
import pandas as pd
import torch
import joblib
import glob

# import your pipeline functions
from pipeline import (
    extract_frames,
    extract_audio,
    extract_audio_features,
    extract_frame_features,
    run_ocr,
)

st.set_page_config(page_title="Valorant Highlights", layout="wide")
st.title("🎥 Valorant Highlights Extractor")

video_file = st.file_uploader("Upload a video (.mp4/.webm)", type=["mp4","webm"])
segment_len = st.slider("Segment length (seconds)", 5, 30, 10)

if video_file and st.button("Detect Highlights"):
    tmp = tempfile.mkdtemp()
    try:
        # 1) save upload
        vid_path = os.path.join(tmp, video_file.name)
        with open(vid_path, "wb") as f:
            f.write(video_file.getbuffer())

        # 2) segment video
        seg_dir = os.path.join(tmp, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        st.text("⏳ Segmenting video…")
        subprocess.run([
            "ffmpeg","-hide_banner","-loglevel","error",
            "-i", vid_path,
            "-c:v","libvpx","-deadline","realtime","-cpu-used","5","-threads","1",
            "-c:a","libopus","-b:a","64k",
            "-f","segment","-segment_time", str(segment_len),
            "-reset_timestamps","1",
            os.path.join(seg_dir, "seg_%04d.webm")
        ], check=True)

        # 2b) wrap into a dummy subdir so pipeline.extract_* sees it
        wrapped = os.path.join(tmp, "wrapped_segments")
        os.makedirs(wrapped, exist_ok=True)
        dummy = os.path.join(wrapped, "all")
        os.makedirs(dummy, exist_ok=True)
        for f in glob.glob(os.path.join(seg_dir, "*.webm")):
            shutil.copy2(f, dummy)

        # 3) extract everything & compute features
        frames_dir = os.path.join(tmp, "frames")
        audio_dir  = os.path.join(tmp, "audio")

        st.text("⏳ Extracting frames…")
        extract_frames(input_dir=wrapped, output_dir=frames_dir, fps=1)

        st.text("⏳ Extracting audio…")
        extract_audio(input_dir=wrapped, output_dir=audio_dir)

        st.text("⏳ Computing audio features…")
        audio_feat = os.path.join(tmp, "audio.parquet")
        extract_audio_features(input_dir=audio_dir,
                               output_file=audio_feat)

        st.text("⏳ Computing frame features…")
        frame_feat = os.path.join(tmp, "frame.parquet")
        extract_frame_features(input_dir=frames_dir,
                               output_file=frame_feat)

        st.text("⏳ Running OCR…")
        ocr_feat   = os.path.join(tmp, "ocr.parquet")
        run_ocr(frames_dir=frames_dir,
                output_file=ocr_feat)

        # 4) load & merge features
        st.text("⏳ Merging features…")
        af = pd.read_parquet(audio_feat)
        ff = pd.read_parquet(frame_feat)
        of = pd.read_parquet(ocr_feat)

        df = af.merge(ff, on="segment_id").merge(of, on="segment_id")
        # drop raw OCR text before inference
        if "ocr_text" in df.columns:
            df = df.drop(columns=["ocr_text"])

        X = df.drop(columns=["segment_id"]).values
        Xt = torch.tensor(X, dtype=torch.float32)

        # 5) load model + label encoder
        st.text("⏳ Loading model…")
        raw = torch.load("highlight_classifier.pth", map_location="cpu")
        if isinstance(raw, dict) and "model" in raw and "le" in raw:
            model_state, le = raw["model"], raw["le"]
        else:
        # old-style checkpoint: just the weight dict → load encoder via joblib
            model_state = raw
            le = joblib.load("label_encoder.pkl")

        # --- rebuild your MLP with the *correct* input size from the checkpoint ---
        # find the first linear layer’s weight tensor in the state dict
        # (we know it lives under key "0.weight" since we used Sequential)
        w0 = model_state["0.weight"]  # torch.Size([hidden_dim, input_dim])
        input_dim, hidden_dim = w0.shape[1], w0.shape[0]
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(le.classes_))
            )
        model.load_state_dict(model_state)
        model.eval()

        # 6) inference
        st.text("⏳ Predicting…")
        with torch.no_grad():
            preds = model(Xt).argmax(dim=1).numpy()
        df["pred"] = le.inverse_transform(preds)

        # 7) pick highlight classes
        HIGHLIGHT_CLASSES = ["ACE","Multi_K","clutch"]
        out = df[df["pred"].isin(HIGHLIGHT_CLASSES)]

        if out.empty:
            st.warning("No highlights found.")

        else:
            from collections import Counter
            counts = Counter(out["pred"])
            total = len(out)
            # build a little bullet-list of “3×ACE, 2×clutch, …”
            summary_lines = "\n".join(f"- {cnt}×{label}" for label, cnt in counts.items())
            st.markdown("## 🗂️ Overall Highlight Summary")
            st.markdown(f"Processed **{total}** highlight segments:\n\n{summary_lines}")
            st.success(f"Found {len(out)} segments:")
            st.dataframe(out[["segment_id","pred"]])

            # 8) stitch highlights
            concat_file = os.path.join(tmp, "to_concat.txt")
            with open(concat_file, "w") as f:
                for sid in out["segment_id"]:
                    f.write(f"file '{os.path.join(seg_dir, sid + '.webm')}'\n")

            highlights = os.path.join(tmp, "highlights.webm")
            subprocess.run([
                "ffmpeg","-hide_banner","-loglevel","error",
                "-f","concat","-safe","0","-i", concat_file,
                "-c","copy", highlights
            ], check=True)

            st.video(highlights)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
