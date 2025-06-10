import os
import glob
from moviepy.audio.io.AudioFileClip import AudioFileClip

# Determine project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Configuration
RAW_DIR     = os.path.join(PROJECT_ROOT, "raw_videos")
AUDIO_DIR   = os.path.join(PROJECT_ROOT, "audio")
SAMPLE_RATE = 16000  # 16 kHz mono

def extract_audio():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for video_path in glob.glob(os.path.join(RAW_DIR, "*.*")):
        base     = os.path.splitext(os.path.basename(video_path))[0]
        out_file = os.path.join(AUDIO_DIR, f"{base}.wav")

        # Load the audio directly
        clip = AudioFileClip(video_path)
        if clip is None:
            print(f"Couldn’t load audio from {video_path}")
            continue

        # Write the audio clip to disk
        clip.write_audiofile(out_file, fps=SAMPLE_RATE, codec="pcm_s16le")
        clip.close()
        print(f"Extracted audio: {out_file}")

if __name__ == "__main__":
    extract_audio()
