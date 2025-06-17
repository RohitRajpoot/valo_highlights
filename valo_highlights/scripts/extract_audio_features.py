import os, glob
import numpy as np
import librosa

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

RAW_AUDIO = os.path.join(PROJECT_ROOT, "audio")
OUT_DIR   = os.path.join(PROJECT_ROOT, "models", "audio_feats")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 16000        # sample rate
N_MELS = 64      # number of Mel bands

def extract():
    for wav_path in sorted(glob.glob(f"{RAW_AUDIO}/*.wav")):
        base = os.path.splitext(os.path.basename(wav_path))[0]
        y, _ = librosa.load(wav_path, sr=SR)
        # Compute a Mel‐spectrogram [n_mels x T]
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_mels=N_MELS
        )
        # Convert to log scale (dB)
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
        out_path = os.path.join(OUT_DIR, f"{base}.npy")
        np.save(out_path, logmelspec)
        print(f"Saved audio features for {base} → {out_path}")

if __name__ == "__main__":
    extract()
