import os
import re
import pandas as pd
import torch
import librosa
import cv2
import numpy as np
from transformers import AutoTokenizer, ViTModel, ViTImageProcessor
import config

def extract_video(path, vit_proc, vit_model, device):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, config.MAX_FRAMES, dtype=int)
        for i in range(total):
            ret, frame = cap.read()
            if not ret: break
            if i in idxs:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    while len(frames) < config.MAX_FRAMES:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
    inp = vit_proc(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vit_model(**inp)
    return out.last_hidden_state[:, 0, :].cpu()   # [30, 768]

def extract_audio_mel(path):
    try:
        y, sr = librosa.load(path, sr=16000)
    except Exception:
        y, sr = np.zeros(16000), 16000
        
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS, hop_length=160)
    mel = librosa.power_to_db(mel).T              # [T, 80]
    mel = torch.tensor(mel, dtype=torch.float32)
    T = mel.shape[0]
    
    if T >= config.MAX_AUDIO_FRAMES:
        mel = mel[:config.MAX_AUDIO_FRAMES]
    else:
        mel = torch.cat([mel, torch.zeros(config.MAX_AUDIO_FRAMES - T, config.N_MELS)], dim=0)
    return mel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read Excel
    df = pd.read_excel(config.EXCEL_PATH)
    df["label_orig"] = df["label"].apply(config.normalize_label)
    df["label_merged"] = df["label"].apply(config.merged_label)

    merged_classes = sorted(df["label_merged"].dropna().unique().tolist())
    label_map = {c: i for i, c in enumerate(merged_classes)}
    print(f"{len(label_map)} merged classes: {label_map}")
    torch.save(label_map, os.path.join(config.OUTPUT_DIR, "label_map.pt"))

    # Load Models
    print("Loading ViT and MuRIL...")
    vit_proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    print("Extracting features...")
    skipped = 0
    for i, row in df.iterrows():
        label = row["label_merged"]
        if label not in label_map:
            skipped += 1; continue

        sid = re.sub(r'\.(mp4|wav|mp3)$', '', str(row["file_id"]).strip())
        vpath = os.path.join(config.VIDEO_DIR, f"{sid}.mp4")
        apath = os.path.join(config.AUDIO_DIR, f"{sid}.mp3")

        if not os.path.exists(vpath) or not os.path.exists(apath):
            print(f"  SKIP missing: {sid}")
            skipped += 1; continue

        print(f"  [{i+1}/{len(df)}] {sid}")
        v_feat = extract_video(vpath, vit_proc, vit_model, device)
        a_feat = extract_audio_mel(apath)

        combined = f"{row['hindi_text']} | {row['hinglish_text']}"

        t_masked = tokenizer(
            f"{combined} [SEP] [MASK]",
            max_length=config.MAX_TEXT_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        t_unmasked = tokenizer(
            f"{combined} [SEP] {label}",
            max_length=config.MAX_TEXT_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        torch.save({
            "label": torch.tensor(label_map[label], dtype=torch.long),
            "t_masked_input_ids": t_masked["input_ids"].squeeze(0),
            "t_masked_attention_mask": t_masked["attention_mask"].squeeze(0),
            "t_unmasked_input_ids": t_unmasked["input_ids"].squeeze(0),
            "t_unmasked_attention_mask": t_unmasked["attention_mask"].squeeze(0),
            "v_features": v_feat,
            "a_features": a_feat,
        }, os.path.join(config.OUTPUT_DIR, f"{sid}.pt"))

    print(f"Done. Skipped {skipped}.")

if __name__ == "__main__":
    main()