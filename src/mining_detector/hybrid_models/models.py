"""Five small deep-learning models for mining detection, written in PyTorch.

Each model is kept under about 50,000 parameters so it trains on a CPU in minutes.
Every model also has a `lite=True` version: half the channel width and a single
attention or LSTM layer where the full model has two.

Input is a multi-band image patch of shape (batch, channels, H, W). The default patch
size is 64 x 64 pixels, the size used by the Open-Pit tiling in openpit_dataset.py.

Two kinds of output:
    segmentation models  -> per-pixel logits of shape (batch, 1, H, W)
    classification models -> one logit per class, shape (batch, n_classes)
    siamese model         -> takes two dates (x_t0, x_t1), returns (batch, n_classes)

Models
------
UNetTransformer       U-shaped segmentation net. A transformer runs on the bottleneck,
                      so every pixel can use context from the whole patch.
ResNetAttentionUNet   U-shaped segmentation net with residual blocks and attention gates.
                      An attention gate lets the decoder focus on useful skip features.
YoloViTLSTM           Patch classifier. A small CNN turns the patch into a grid of cells
                      (the YOLO-style idea), a transformer mixes the cells, and an LSTM
                      reads them as a sequence.
CNNBiLSTM             Patch classifier. A CNN extracts features, then a bidirectional LSTM
                      reads the feature map row by row.
SiameseCNNTransformer Two-date change classifier. One shared CNN encodes both dates, the
                      difference between the two encodings goes through a transformer,
                      and a head decides "new disturbance" or "no change".
"""

from __future__ import annotations

import torch
from torch import nn

SEGMENTATION_MODELS = ("UNetTransformer", "ResNetAttentionUNet")
CLASSIFICATION_MODELS = ("YoloViTLSTM", "CNNBiLSTM")
SIAMESE_MODELS = ("SiameseCNNTransformer",)
ALL_MODELS = SEGMENTATION_MODELS + CLASSIFICATION_MODELS + SIAMESE_MODELS


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class _TransformerStack(nn.Module):
    """Transformer encoder over a sequence of tokens.

    No learned position embedding is used. The convolutions before it already encode
    where each token sits, and dropping the embedding keeps the models small.
    """

    def __init__(self, dim: int, n_tokens: int, n_layers: int, heads: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim,
                                           batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


# ----------------------------------------------------------------- segmentation --

class UNetTransformer(nn.Module):
    """U-Net with a transformer at the bottleneck.

    Full: width 8, two transformer layers. Lite: width 4, one transformer layer.
    The bottleneck of a 64 x 64 input is 16 x 16 = 256 tokens.
    """

    def __init__(self, in_ch: int = 6, n_classes: int = 1, lite: bool = False, image_size: int = 64):
        super().__init__()
        w = 4 if lite else 8
        n_layers = 1 if lite else 2
        bottom = image_size // 4
        self.enc1 = _conv_block(in_ch, w)
        self.enc2 = _conv_block(w, 2 * w)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(2 * w, 4 * w)
        self.bottom = bottom
        self.transformer = _TransformerStack(4 * w, bottom * bottom, n_layers)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = _conv_block(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = _conv_block(2 * w, w)
        self.head = nn.Conv2d(w, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        B, C, H, W = b.shape
        tokens = b.flatten(2).transpose(1, 2)          # (B, H*W, C)
        tokens = self.transformer(tokens)
        b = tokens.transpose(1, 2).reshape(B, C, H, W)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class _ResidualBlock(nn.Module):
    """Two convolutions with a skip connection, so gradients flow through deep stacks."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(c_in), nn.ReLU(inplace=True), nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.Conv2d(c_out, c_out, 3, padding=1),
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class AttentionGate(nn.Module):
    """Attention gate (Oktay et al., 2018).

    The decoder signal g decides which skip-connection features x matter. The gate outputs
    a weight between 0 and 1 for every pixel, and the skip features are multiplied by it.
    """

    def __init__(self, c_skip: int, c_gate: int, c_inter: int):
        super().__init__()
        self.w_x = nn.Conv2d(c_skip, c_inter, 1)
        self.w_g = nn.Conv2d(c_gate, c_inter, 1)
        self.psi = nn.Sequential(nn.Conv2d(c_inter, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        attn = self.psi(torch.relu(self.w_x(x) + self.w_g(g)))
        return x * attn


class ResNetAttentionUNet(nn.Module):
    """U-Net with residual blocks and attention gates on each skip connection.

    Full: width 8. Lite: width 4 (the attention gates stay, since they are already small).
    """

    def __init__(self, in_ch: int = 6, n_classes: int = 1, lite: bool = False):
        super().__init__()
        w = 4 if lite else 8
        self.enc1 = _ResidualBlock(in_ch, w)
        self.enc2 = _ResidualBlock(w, 2 * w)
        self.enc3 = _ResidualBlock(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.gate2 = AttentionGate(2 * w, 2 * w, max(1, w))
        self.dec2 = _ResidualBlock(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.gate1 = AttentionGate(w, w, max(1, w // 2))
        self.dec1 = _ResidualBlock(2 * w, w)
        self.head = nn.Conv2d(w, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([self.gate2(e2, d2), d2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.gate1(e1, d1), d1], dim=1))
        return self.head(d1)


# ----------------------------------------------------------------- classification --

class YoloViTLSTM(nn.Module):
    """Patch classifier: grid of cells, then transformer, then LSTM.

    A CNN with two stride-2 steps turns a 64 x 64 patch into a 16 x 16 grid of cells.
    Each cell is one token. Full: two transformer layers and a two-layer LSTM.
    Lite: one transformer layer and a one-layer LSTM.
    """

    def __init__(self, in_ch: int = 6, n_classes: int = 2, lite: bool = False, image_size: int = 64):
        super().__init__()
        d = 16 if lite else 32
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, d // 2, 3, stride=2, padding=1), nn.BatchNorm2d(d // 2), nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, 3, stride=2, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        cells = (image_size // 4) ** 2
        self.transformer = _TransformerStack(d, cells, 1 if lite else 2)
        self.lstm = nn.LSTM(d, d, num_layers=1 if lite else 2, batch_first=True)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).flatten(2).transpose(1, 2)   # (B, cells, d)
        feats = self.transformer(feats)
        seq, _ = self.lstm(feats)
        return self.head(seq.mean(dim=1))


class CNNBiLSTM(nn.Module):
    """Patch classifier: CNN features read row by row by a bidirectional LSTM.

    Full: two LSTM layers. Lite: one LSTM layer and half the CNN width.
    """

    def __init__(self, in_ch: int = 6, n_classes: int = 2, lite: bool = False, image_size: int = 64):
        super().__init__()
        c = 8 if lite else 16
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        grid = image_size // 4
        hidden = 16
        self.lstm = nn.LSTM(c * grid, hidden, num_layers=1 if lite else 2,
                            batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                                 # (B, c, H', W')
        B, C, H, W = f.shape
        rows = f.permute(0, 2, 1, 3).reshape(B, H, C * W)   # each image row is one time step
        seq, _ = self.lstm(rows)
        return self.head(seq[:, -1])


class SiameseCNNTransformer(nn.Module):
    """Two-date change classifier with one shared encoder.

    Both dates go through the same CNN. The absolute difference of the two feature maps
    highlights what changed. A transformer then reads the changed-feature tokens.
    Full: two transformer layers. Lite: one layer and half the width.
    The output is two logits: 0 = no new disturbance, 1 = new disturbance.
    """

    def __init__(self, in_ch: int = 6, n_classes: int = 2, lite: bool = False, image_size: int = 64):
        super().__init__()
        d = 16 if lite else 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, d // 2, 3, stride=2, padding=1), nn.BatchNorm2d(d // 2), nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, 3, stride=2, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        cells = (image_size // 4) ** 2
        self.transformer = _TransformerStack(d, cells, 1 if lite else 2)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor) -> torch.Tensor:
        f0 = self.encoder(x_t0).flatten(2).transpose(1, 2)
        f1 = self.encoder(x_t1).flatten(2).transpose(1, 2)
        diff = (f1 - f0).abs()
        out = self.transformer(diff)
        return self.head(out.mean(dim=1))


# ------------------------------------------------------------------- registry --

_BUILDERS = {
    "UNetTransformer": UNetTransformer,
    "ResNetAttentionUNet": ResNetAttentionUNet,
    "YoloViTLSTM": YoloViTLSTM,
    "CNNBiLSTM": CNNBiLSTM,
    "SiameseCNNTransformer": SiameseCNNTransformer,
}


def build_model(name: str, in_ch: int = 6, lite: bool = False, image_size: int = 64) -> nn.Module:
    """Create a model by name. Segmentation models take n_classes=1 (one mask channel)."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(ALL_MODELS)}.")
    builder = _BUILDERS[name]
    if name in SEGMENTATION_MODELS:
        if name == "UNetTransformer":
            return builder(in_ch=in_ch, n_classes=1, lite=lite, image_size=image_size)
        return builder(in_ch=in_ch, n_classes=1, lite=lite)
    if name in CLASSIFICATION_MODELS or name in SIAMESE_MODELS:
        return builder(in_ch=in_ch, n_classes=2, lite=lite, image_size=image_size)
    raise ValueError(name)  # unreachable, the check above covers all names
