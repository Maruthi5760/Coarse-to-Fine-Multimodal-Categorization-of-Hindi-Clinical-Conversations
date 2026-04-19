import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

import config
from dataset import HindiIntentDataset
from model import HindiMVCL_DAF

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = torch.load(os.path.join(config.OUTPUT_DIR, "label_map.pt"))
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    model = HindiMVCL_DAF(num_classes=num_classes).to(device)
    model.load_state_dict(
        torch.load(os.path.join(config.MODEL_SAVE_DIR, "best_hindi_intent.pth"), map_location=device), 
        strict=False
    )
    model.eval()

    full_ds = HindiIntentDataset(config.OUTPUT_DIR, augment=False)
    test_idx = torch.load(os.path.join(config.MODEL_SAVE_DIR, "test_indices_v2.pt"))
    loader = DataLoader(Subset(full_ds, test_idx), batch_size=16, shuffle=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _, _ = model(
                batch["ids_l"].to(device), batch["mask_l"].to(device),
                batch["ids_m"].to(device), batch["mask_m"].to(device),
                batch["video"].to(device), batch["audio"].to(device)
            )
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_true.extend(batch["label"].cpu().tolist())

    for t, p in zip(all_true, all_preds):
        marker = "✅" if t == p else "❌"
        print(f"{marker}  True: {idx_to_label[t]:<15} | Pred: {idx_to_label[p]}")

    correct = sum(p == t for p, t in zip(all_preds, all_true))
    print(f"\n{'='*50}\nAccuracy : {correct / len(all_true) * 100:.2f}%  ({correct}/{len(all_true)})\n{'='*50}\n")

    used = sorted(set(all_true + all_preds))
    print(classification_report(all_true, all_preds, labels=used, target_names=[idx_to_label[i] for i in used], zero_division=0))

if __name__ == "__main__":
    main()