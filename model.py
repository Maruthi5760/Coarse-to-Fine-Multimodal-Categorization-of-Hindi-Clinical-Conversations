import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class InfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, anchor, positive):
        B = anchor.shape[0]
        if B < 2: return torch.tensor(0.0, device=anchor.device)
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        logits = a @ p.T / self.tau
        labels = torch.arange(B, device=a.device)
        return F.cross_entropy(logits, labels)

class ProtoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, h, labels):
        if h.shape[0] < 2: return torch.tensor(0.0, device=h.device)
        h = F.normalize(h, dim=-1)
        classes = labels.unique()
        protos = torch.stack([
            F.normalize(h[labels == c].mean(0), dim=-1)
            for c in classes
        ])
        lbl_map = {c.item(): i for i, c in enumerate(classes)}
        y_idx = torch.tensor([lbl_map[y.item()] for y in labels], device=h.device)
        logits = h @ protos.T / self.tau
        return F.cross_entropy(logits, y_idx)

class LabelSmoothCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.s = smoothing; self.w = weight; self.C = num_classes

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.w)
        smooth = -F.log_softmax(logits, dim=-1).mean(-1).mean()
        return (1 - self.s) * ce + self.s * smooth

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=80, d_model=256):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.gru = nn.GRU(d_model, d_model // 2, bidirectional=True, batch_first=True, num_layers=2, dropout=0.1)

    def forward(self, x):
        x = F.relu(self.proj(x))
        out, _ = self.gru(x)
        return out

class VisualEncoder(nn.Module):
    def __init__(self, d_model=256, vis_in=768):
        super().__init__()
        self.proj = nn.Linear(vis_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        x = F.relu(self.proj(x))
        return self.enc(x)

class CoarseAttn(nn.Module):
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def _align(self, x, L):
        return F.interpolate(x.permute(0, 2, 1), size=L, mode="linear", align_corners=False).permute(0, 2, 1)

    def forward(self, M_tm, M_v, M_a):
        L_t = M_tm.shape[1]
        M_v = self._align(M_v, L_t)
        M_a = self._align(M_a, L_t)
        ctx, _ = self.attn(query=M_tm, key=M_v, value=M_a)
        M_c = self.norm(M_tm + ctx)
        M_c = M_c + self.ff(M_c)
        return M_c, M_c.mean(dim=1)

class DAF(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, fine, coarse_vec):
        exp = coarse_vec.unsqueeze(1).expand_as(fine)
        g = self.gate(torch.cat([fine, exp], dim=-1))
        return g * fine + (1 - g) * exp

class HindiMVCL_DAF(nn.Module):
    def __init__(self, num_classes, d_model=256, n_mels=80, vis_in=768, class_weights=None):
        super().__init__()
        D = d_model
        self.text_encoder = AutoModel.from_pretrained("google/muril-base-cased")
        
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < 10:
                for p in layer.parameters(): p.requires_grad = False
                
        self.text_proj = nn.Linear(768, D)
        self.audio_enc = AudioEncoder(n_mels=n_mels, d_model=D)
        self.visual_enc = VisualEncoder(d_model=D, vis_in=vis_in)
        self.coarse = CoarseAttn(d_model=D, nhead=4)
        self.daf_fine = DAF(D)
        self.daf_coarse = DAF(D)
        self.classifier = nn.Sequential(
            nn.LayerNorm(D), nn.Dropout(0.4), nn.Linear(D, D // 2),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(D // 2, num_classes),
        )
        self.infonce = InfoNCE(tau=0.1)
        self.proto_nce = ProtoNCE(tau=0.1)
        self.ce_loss = LabelSmoothCE(num_classes, smoothing=0.1, weight=class_weights)

    def _encode_text(self, ids, mask):
        out = self.text_encoder(input_ids=ids, attention_mask=mask)
        return F.gelu(self.text_proj(out.last_hidden_state))

    def forward(self, ids_l, mask_l, ids_m, mask_m, video, audio, labels=None):
        M_tl = self._encode_text(ids_l, mask_l)
        M_tm = self._encode_text(ids_m, mask_m)
        M_v = self.visual_enc(video)
        M_a = self.audio_enc(audio)

        h_tl, h_tm, h_v, h_a = M_tl.mean(1), M_tm.mean(1), M_v.mean(1), M_a.mean(1)
        M_c, h_c = self.coarse(M_tm, M_v, M_a)
        M_f = self.daf_fine(M_tm, h_c)
        M_cf = self.daf_coarse(M_c, h_c)

        h_f = M_f.mean(1)
        h_cls = M_cf[:, 0, :]
        logits = self.classifier(h_cls)

        loss = None
        if labels is not None:
            L_ce = self.ce_loss(logits, labels)
            L_mv = (self.infonce(h_tl, h_tm) + self.infonce(h_tl, h_v) + self.infonce(h_tl, h_a) + self.infonce(h_tl, h_f)) / 4.0
            L_proto = self.proto_nce(h_cls, labels)
            loss = L_ce + 0.15 * L_mv + 0.1 * L_proto

        return logits, h_cls, loss