"""Five hybrid deep learning architectures for mining detection.

Two data regimes are used, matched to what each architecture actually
needs and what ground truth actually exists:

- U-Net+Transformer and ResNet+Attention U-Net are genuine per-pixel
  segmentation models, trained against real binary masks from the Open-Pit
  Mining Areas Dataset (openpit_dataset.py, 10 real image/mask tiles,
  64x64 patches). `forward(x, return_map=True)` returns the full-resolution
  segmentation logit map for this; the default `return_map=False` path
  (global-average-pooled classification logit) is kept for the small
  hard-case chip set (dataset.py) where no masks exist.
- YOLO+ViT+LSTM, CNN+BiLSTM, and Siamese CNN+Transformer are
  classification-only (YOLO/LSTM don't have a natural per-pixel output;
  the Siamese model is genuinely bi-temporal but the only bi-temporal
  pairs available are the 32 hard-case chips, not the single-date Open-Pit
  dataset). YOLO+ViT+LSTM and CNN+BiLSTM ARE trained on the Open-Pit
  dataset's patch-level labels (much larger n than the 32 chips), giving
  them a real accuracy signal rather than a 24-example proof of concept.

Every architecture is a genuine instance of its named components -- real
conv backbones, real multi-head self-attention (nn.TransformerEncoderLayer),
real LSTM/BiLSTM, a real Siamese shared-weight encoder, real residual
blocks, a real attention-gate mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


# --------------------------------------------------------------------------
# 1. YOLO + Vision Transformer + LSTM
# --------------------------------------------------------------------------
class YoloViTLSTM(nn.Module):
    """Darknet-style conv backbone (YOLO's feature extractor) -> transformer
    encoder over the resulting spatial tokens (ViT-style global attention)
    -> LSTM over that token sequence -> classification head.
    """

    def __init__(self, embed_dim: int = 32, backbone_ch: int = 16, vit_layers: int = 2,
                 vit_heads: int = 4, lstm_hidden: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            _conv_block(3, backbone_ch),        # 64 -> 32
            _conv_block(backbone_ch, backbone_ch * 2),   # 32 -> 16
            _conv_block(backbone_ch * 2, embed_dim),     # 16 -> 8
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=vit_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.vit = nn.TransformerEncoder(encoder_layer, num_layers=vit_layers)
        self.lstm = nn.LSTM(embed_dim, hidden_size=lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        feat = self.backbone(x)                        # (B, C, 8, 8)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)        # (B, 64, C)
        tokens = self.vit(tokens)                       # (B, 64, C)
        _, (h_n, _) = self.lstm(tokens)                 # h_n: (1, B, 32)
        return self.head(h_n.squeeze(0)).squeeze(-1)


# --------------------------------------------------------------------------
# 2. U-Net + Transformer
# --------------------------------------------------------------------------
class UNetTransformer(nn.Module):
    """U-Net encoder-decoder with a transformer bottleneck. The decoder's
    output map is global-average-pooled into the classification decision
    (no pixel masks exist to supervise it as true segmentation), so it is
    still an active part of the forward pass rather than dead weight.
    """

    def __init__(self, embed_dim: int = 32, enc1_ch: int = 16, transformer_layers: int = 2,
                 transformer_heads: int = 4):
        super().__init__()
        self.enc1 = _conv_block(3, enc1_ch, stride=2)   # 64 -> 32
        self.enc2 = _conv_block(enc1_ch, embed_dim, stride=2)  # 32 -> 16

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=transformer_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.bottleneck_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.up1 = nn.ConvTranspose2d(embed_dim, enc1_ch, kernel_size=2, stride=2)  # 16 -> 32
        self.dec1 = _conv_block(enc1_ch * 2, enc1_ch, stride=1)  # concat with enc1 skip
        self.up2 = nn.ConvTranspose2d(enc1_ch, enc1_ch // 2, kernel_size=2, stride=2)  # 32 -> 64
        self.dec2 = nn.Conv2d(enc1_ch // 2, 1, kernel_size=1)

        self.head = nn.Linear(1, 1)

    def forward(self, x, return_map: bool = False):
        e1 = self.enc1(x)          # (B, 16, 32, 32)
        e2 = self.enc2(e1)         # (B, C, 16, 16)

        B, C, H, W = e2.shape
        tokens = e2.flatten(2).transpose(1, 2)
        tokens = self.bottleneck_transformer(tokens)
        bottleneck = tokens.transpose(1, 2).reshape(B, C, H, W)

        d1 = self.up1(bottleneck)              # (B, 16, 32, 32)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d2 = self.up2(d1)                      # (B, 8, 64, 64)
        seg_map = self.dec2(d2)                # (B, 1, 64, 64)

        if return_map:
            return F.interpolate(seg_map, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        pooled = seg_map.mean(dim=[2, 3])      # (B, 1) -- global average pool
        return self.head(pooled).squeeze(-1)


# --------------------------------------------------------------------------
# 3. CNN + BiLSTM
# --------------------------------------------------------------------------
class CNNBiLSTM(nn.Module):
    """CNN spatial feature extractor, then a BiLSTM run over the feature
    map's rows as a sequence (models row-wise spatial dependencies,
    complementing the CNN's local receptive field).
    """

    def __init__(self, cnn_channels: int = 32, lstm_hidden: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_block(3, 16),
            _conv_block(16, cnn_channels),
        )
        self.bilstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        feat = self.cnn(x)                     # (B, C, H, W) e.g. 16x16
        B, C, H, W = feat.shape
        # Each row of the feature map is one timestep; average across width
        # to get a per-row feature vector.
        rows = feat.mean(dim=3).transpose(1, 2)  # (B, H, C)
        _, (h_n, _) = self.bilstm(rows)
        out = torch.cat([h_n[0], h_n[1]], dim=1)  # forward + backward final states
        return self.head(out).squeeze(-1)


# --------------------------------------------------------------------------
# 4. Siamese CNN + Transformer
# --------------------------------------------------------------------------
class SiameseCNNTransformer(nn.Module):
    """A shared-weight ('Siamese') CNN encodes the t0 and t1 chips
    independently into a feature vector each; a transformer encoder then
    attends across the resulting 2-token sequence [before, after], modeling
    their relationship, before a classification head. This is the one
    architecture using real bi-temporal input (paired t0/t1 crops at the
    same location), matching this project's O3 change-detection framing.
    """

    def __init__(self, embed_dim: int = 32, backbone_ch: int = 16, transformer_layers: int = 2,
                 transformer_heads: int = 4):
        super().__init__()
        self.shared_cnn = nn.Sequential(
            _conv_block(3, backbone_ch),
            _conv_block(backbone_ch, embed_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=transformer_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x0, x1):
        f0 = self.shared_cnn(x0).flatten(1)   # (B, C) -- t0 embedding
        f1 = self.shared_cnn(x1).flatten(1)   # (B, C) -- t1 embedding, SAME weights
        tokens = torch.stack([f0, f1], dim=1)  # (B, 2, C)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# --------------------------------------------------------------------------
# 5. ResNet + Attention U-Net
# --------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class AttentionGate(nn.Module):
    """Attention U-Net's gating mechanism: the decoder's (coarser, more
    semantic) signal gates the encoder's skip connection before it's
    concatenated in, suppressing irrelevant background regions."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, kernel_size=1)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, kernel_size=1)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.sigmoid(self.psi(self.act(g + s)))
        return skip * attn


class ResNetAttentionUNet(nn.Module):
    """ResNet-style encoder (conv stem + residual blocks) feeding an
    Attention U-Net decoder. As with UNetTransformer, no pixel masks exist
    to supervise true segmentation, so the decoder's output map is
    global-average-pooled into the classification decision.
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        self.stem = _conv_block(3, base_ch, stride=2)       # 64 -> 32
        self.res1 = ResidualBlock(base_ch)
        self.down = _conv_block(base_ch, base_ch * 2, stride=2)  # 32 -> 16
        self.res2 = ResidualBlock(base_ch * 2)

        self.up = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)  # 16 -> 32
        self.attn_gate = AttentionGate(base_ch, base_ch, base_ch // 2)
        self.dec = _conv_block(base_ch * 2, base_ch, stride=1)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        self.head = nn.Linear(1, 1)

    def forward(self, x, return_map: bool = False):
        s1 = self.res1(self.stem(x))       # (B, base_ch, 32, 32) -- skip connection
        bottleneck = self.res2(self.down(s1))  # (B, 2*base_ch, 16, 16)

        up = self.up(bottleneck)           # (B, base_ch, 32, 32)
        gated_skip = self.attn_gate(gate=up, skip=s1)
        d = self.dec(torch.cat([up, gated_skip], dim=1))
        seg_map = self.out_conv(d)         # (B, 1, 32, 32)

        if return_map:
            return F.interpolate(seg_map, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        pooled = seg_map.mean(dim=[2, 3])
        return self.head(pooled).squeeze(-1)


MODEL_REGISTRY = {
    "yolo_vit_lstm": YoloViTLSTM,
    "unet_transformer": UNetTransformer,
    "cnn_bilstm": CNNBiLSTM,
    "siamese_cnn_transformer": SiameseCNNTransformer,
    "resnet_attention_unet": ResNetAttentionUNet,
}

SIAMESE_MODELS = {"siamese_cnn_transformer"}

# --------------------------------------------------------------------------
# Lightweight variants -- addresses MineFormer's and TransSiamUNet's stated
# future work ("further reduction of computational cost ... for real-time
# or edge-device deployment"). Each halves embedding width and drops to a
# single attention/LSTM layer where applicable, rather than an arbitrary
# shrink -- a controlled comparison of the SAME architecture family at two
# scales, matching how MineFormer and the "When Less Is More" paper (also
# cited) frame the efficiency question.
LITE_KWARGS = {
    "yolo_vit_lstm": dict(embed_dim=16, backbone_ch=8, vit_layers=1, vit_heads=2, lstm_hidden=16),
    "unet_transformer": dict(embed_dim=16, enc1_ch=8, transformer_layers=1, transformer_heads=2),
    "cnn_bilstm": dict(cnn_channels=16, lstm_hidden=16),
    "siamese_cnn_transformer": dict(embed_dim=16, backbone_ch=8, transformer_layers=1, transformer_heads=2),
    "resnet_attention_unet": dict(base_ch=8),
}


def build_model(name: str, lite: bool = False) -> nn.Module:
    """Constructs a model by registry name, either full-size (default
    constructor args) or the lite variant (LITE_KWARGS)."""
    cls = MODEL_REGISTRY[name]
    if lite:
        return cls(**LITE_KWARGS[name])
    return cls()
