import os
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import config

class HindiIntentDataset(Dataset):
    def __init__(self, features_dir, augment=False):
        self.dir = features_dir
        self.augment = augment
        self.files = sorted([
            f for f in os.listdir(features_dir)
            if f.endswith(".pt") and f != "label_map.pt"
        ])

    def __len__(self): return len(self.files)

    def _fix(self, t, L):
        if t.shape[0] == L: return t
        if t.shape[0] > L: return t[:L]
        pad_shape = (L - t.shape[0],) + t.shape[1:]
        return torch.cat([t, torch.zeros(pad_shape)], dim=0)

    def _fix1d(self, t, L):
        if t.shape[0] == L: return t
        if t.shape[0] > L: return t[:L]
        return F.pad(t, (0, L - t.shape[0]), value=0)

    def _aug_audio(self, a):
        if random.random() > 0.5:
            a = a + torch.randn_like(a) * 0.01
        if random.random() > 0.7:
            T = a.shape[0]
            ml = max(1, int(T * 0.1))
            s = random.randint(0, T - ml)
            a[s:s+ml] = 0.0
        return a

    def _aug_visual(self, v):
        if random.random() > 0.5:
            v = v + torch.randn_like(v) * 0.005
        return v

    def __getitem__(self, idx):
        d = torch.load(os.path.join(self.dir, self.files[idx]), weights_only=False)

        ids_m = self._fix1d(d["t_masked_input_ids"], config.MAX_TEXT_LEN)
        mask_m = self._fix1d(d["t_masked_attention_mask"], config.MAX_TEXT_LEN)
        ids_l = self._fix1d(d["t_unmasked_input_ids"], config.MAX_TEXT_LEN)
        mask_l = self._fix1d(d["t_unmasked_attention_mask"], config.MAX_TEXT_LEN)

        v = self._fix(d["v_features"], config.MAX_FRAMES)
        a = self._fix(d["a_features"], config.MAX_AUDIO_FRAMES)

        if self.augment:
            v = self._aug_visual(v.clone())
            a = self._aug_audio(a.clone())

        return {
            "ids_l": ids_l, "mask_l": mask_l, 
            "ids_m": ids_m, "mask_m": mask_m, 
            "video": v, "audio": a, "label": d["label"],
        }

def smart_split(files, labels, test_ratio=0.2, seed=42):
    random.seed(seed)
    class_to_idx = {}
    for i, l in enumerate(labels):
        class_to_idx.setdefault(l, []).append(i)
    
    train_idx, test_idx, singletons = [], [], []
    for l, idxs in class_to_idx.items():
        random.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs); singletons.append(l)
        else:
            n = max(1, int(len(idxs) * test_ratio))
            test_idx.extend(idxs[:n]); train_idx.extend(idxs[n:])
            
    if singletons:
        print(f"  Singletons forced to train: {singletons}")
    return train_idx, test_idx