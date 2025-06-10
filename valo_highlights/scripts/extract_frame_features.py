import os, glob
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Compute project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Folders
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames")
FEAT_DIR   = os.path.join(PROJECT_ROOT, "models", "frame_feats")
os.makedirs(FEAT_DIR, exist_ok=True)

# Load ResNet18 without its final layer
device = torch.device("cpu")
resnet = models.resnet18(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()
resnet.eval()

# Preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract():
    for clip in sorted(os.listdir(FRAMES_DIR)):
        frame_paths = sorted(glob.glob(f"{FRAMES_DIR}/{clip}/*.jpg"))
        feats = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            x   = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
            with torch.no_grad():
                f = resnet(x).cpu().numpy().squeeze()      # [512]
            feats.append(f)
        feats = np.stack(feats)                          # [T,512]
        out_path = os.path.join(FEAT_DIR, f"{clip}.npy")
        np.save(out_path, feats)
        print(f"Saved {clip} → {out_path}")

if __name__=="__main__":
    extract()
