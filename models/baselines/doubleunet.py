import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import vgg16, VGG16_Weights
    _VGG_WEIGHTS = VGG16_Weights.DEFAULT
except ImportError:
    from torchvision.models import vgg16
    _VGG_WEIGHTS = True


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, **kwargs): return self.block(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )
    def forward(self, x, **kwargs):
        return x * self.se(x).view(x.shape[0], x.shape[1], 1, 1)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),
                          nn.BatchNorm2d(out_ch), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6),
                          nn.BatchNorm2d(out_ch), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12),
                          nn.BatchNorm2d(out_ch), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18),
                          nn.BatchNorm2d(out_ch), nn.ReLU()),
        ])
        self.pool    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def forward(self, x, **kwargs):
        feats = [c(x) for c in self.convs]
        feats.append(F.interpolate(self.pool(x), size=x.shape[2:],
                                   mode='bilinear', align_corners=True))
        return self.project(torch.cat(feats, dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
        self.se   = SqueezeExcite(out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.se(self.conv(x))


# ---------------------------------------------------------------------------
# DoubleU-Net with VGG16 encoder (Jha et al., 2020)
# ---------------------------------------------------------------------------
class DoubleUNet(nn.Module):
    """
    DoubleU-Net for medical image segmentation.
    Network 1: VGG16 encoder + ASPP + decoder with SE blocks
    Network 2: fresh encoder + ASPP + decoder combining both network skips
    Final output: element-wise product of both network outputs

    VGG16 feature blocks (before each maxpool):
        Block 1: features[0:4]   → 64ch
        Block 2: features[5:9]   → 128ch
        Block 3: features[10:16] → 256ch
        Block 4: features[17:23] → 512ch
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        # ---- VGG16 encoder (Network 1) ----
        vgg   = vgg16(weights=_VGG_WEIGHTS)
        feats = list(vgg.features.children())

        # Extract conv blocks before each maxpool — use as skip sources
        self.vgg_block1 = nn.Sequential(*feats[0:4])    # → 64ch
        self.pool1      = feats[4]
        self.vgg_block2 = nn.Sequential(*feats[5:9])    # → 128ch
        self.pool2      = feats[9]
        self.vgg_block3 = nn.Sequential(*feats[10:16])  # → 256ch
        self.pool3      = feats[16]
        self.vgg_block4 = nn.Sequential(*feats[17:23])  # → 512ch
        self.pool4      = feats[23]

        # Freeze VGG16 weights
        for p in vgg.parameters():
            p.requires_grad = False

        self.aspp1  = ASPP(512, 64)

        self.dec1_1 = DecoderBlock(64,  512, 256)
        self.dec1_2 = DecoderBlock(256, 256, 128)
        self.dec1_3 = DecoderBlock(128, 128, 64)
        self.dec1_4 = DecoderBlock(64,  64,  32)
        self.out1   = nn.Conv2d(32, n_classes, 1)

        # ---- Fresh encoder (Network 2) ----
        self.enc2_1 = ConvBlock(n_channels, 32)
        self.enc2_2 = ConvBlock(32,  64)
        self.enc2_3 = ConvBlock(64,  128)
        self.enc2_4 = ConvBlock(128, 256)
        self.pool   = nn.MaxPool2d(2)

        self.aspp2  = ASPP(256, 64)

        # Decoder 2 — skips from Network 2 + Network 1
        self.dec2_1 = DecoderBlock(64,  256 + 512, 256)
        self.dec2_2 = DecoderBlock(256, 128 + 256, 128)
        self.dec2_3 = DecoderBlock(128, 64  + 128, 64)
        self.dec2_4 = DecoderBlock(64,  32  + 64,  32)
        self.out2   = nn.Conv2d(32, n_classes, 1)

    def _cat(self, a, b):
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=True)
        return torch.cat([a, b], dim=1)

    def forward(self, x, **kwargs):
        # ---- Network 1: VGG16 encoder ----
        s1 = self.vgg_block1(x)
        s2 = self.vgg_block2(self.pool1(s1))
        s3 = self.vgg_block3(self.pool2(s2))
        s4 = self.vgg_block4(self.pool3(s3))

        b1 = self.aspp1(self.pool4(s4))

        d1   = self.dec1_1(b1, s4)
        d2   = self.dec1_2(d1, s3)
        d3   = self.dec1_3(d2, s2)
        d4   = self.dec1_4(d3, s1)
        out1 = torch.sigmoid(self.out1(d4))

        # Mask input for Network 2
        x2 = x * out1

        # ---- Network 2: fresh encoder ----
        e1 = self.enc2_1(x2)
        e2 = self.enc2_2(self.pool(e1))
        e3 = self.enc2_3(self.pool(e2))
        e4 = self.enc2_4(self.pool(e3))

        b2 = self.aspp2(self.pool(e4))

        f1   = self.dec2_1(b2, self._cat(e4, s4))
        f2   = self.dec2_2(f1, self._cat(e3, s3))
        f3   = self.dec2_3(f2, self._cat(e2, s2))
        f4   = self.dec2_4(f3, self._cat(e1, s1))
        out2 = self.out2(f4)

        # Final output
        return out2 * out1