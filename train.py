import os
import torch
import math
from collections import Counter
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR

import config
from dataset import HindiIntentDataset, smart_split
from model import HindiMVCL_DAF

def cosine_warmup(optimizer, warmup, total):
    def fn(step):
        if step < warmup: return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return LambdaLR(optimizer, fn)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_map = torch.load(os.path.join(config.OUTPUT_DIR, "label_map.pt"))
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    full_ds = HindiIntentDataset(config.OUTPUT_DIR, augment=False)
    all_labels = [full_ds[i]["label"].item() for i in range(len(full_ds))]

    print("\nClass distribution (merged):")
    cc = Counter(all_labels)
    for li, cnt in sorted(cc.items(), key=lambda x: x[1]):
        print(f"  [{cnt:3d}] {idx_to_label[li]}")

    train_idx, test_idx = smart_split(full_ds.files, all_labels, test_ratio=0.2, seed=42)
    torch.save(test_idx, os.path.join(config.MODEL_SAVE_DIR, "test_indices_v2.pt"))

    train_ds = HindiIntentDataset(config.OUTPUT_DIR, augment=True)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=config.BATCH_SIZE, shuffle=False)

    train_labels = [all_labels[i] for i in train_idx]
    cc_train = Counter(train_labels)
    weights = torch.zeros(num_classes)
    for i in range(num_classes): weights[i] = 1.0 / cc_train.get(i, 1)
    weights = (weights / weights.sum() * num_classes).to(device)

    model = HindiMVCL_DAF(num_classes=num_classes, class_weights=weights).to(device)

    text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    other_params = [p for p in model.parameters() if p.requires_grad and not any(p is tp for tp in text_params)]

    optimizer = torch.optim.AdamW([
        {"params": text_params, "lr": config.LR},
        {"params": other_params, "lr": config.LR * 5},
    ], weight_decay=0.01)

    total_steps = config.EPOCHS * len(train_loader)
    scheduler = cosine_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_acc, no_improve = 0.0, 0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _, loss = model(
                batch["ids_l"].to(device), batch["mask_l"].to(device), 
                batch["ids_m"].to(device), batch["mask_m"].to(device), 
                batch["video"].to(device), batch["audio"].to(device), 
                batch["label"].to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    logits, _, _ = model(
                        batch["ids_l"].to(device), batch["mask_l"].to(device),
                        batch["ids_m"].to(device), batch["mask_m"].to(device),
                        batch["video"].to(device), batch["audio"].to(device)
                    )
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == batch["label"].to(device)).sum().item()
                    total += batch["label"].size(0)
                    
            acc = correct / total * 100
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.1f}%")

            if acc > best_acc:
                best_acc, no_improve = acc, 0
                torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, "best_hindi_intent.pth"))
                print(f"  ✅ New best: {best_acc:.1f}%")
            else:
                no_improve += 5
                if no_improve >= config.PATIENCE:
                    print("Early stopping triggered."); break
        else:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()