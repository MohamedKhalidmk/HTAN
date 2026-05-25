import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transattunet.unet_parts import DoubleConv, Down, Up, OutConv
from models.transattunet.unet_parts_att_transformer import (
    PAM_Module, PositionEmbeddingLearned, ScaledDotProductAttention
)
from models.transattunet.unet_parts_att_multiscale import MultiConv
from models.htan.mhc import ManifoldConstrainedHyperConnection, ReshapingSAA


# ---------------------------------------------------------------------------
# Official SAA — exactly matches TransAttUNet bottleneck
# Copied from official forward pass:
#   x5_pam  = self.pam(x5)
#   x5_pos  = self.pos(x5)         # pos takes 512//factor channels
#   x5      = x5 + x5_pos
#   x5_sdpa = self.sdpa(x5)
#   x5      = x5_sdpa + x5_pam
# Adds warmup lambda: epoch 0->20, lambda 0->1
# ---------------------------------------------------------------------------
class OfficialSAA(nn.Module):
    def __init__(self, channels):
        """
        channels: actual channel count of the feature map (e.g. 512)
        PositionEmbeddingLearned takes channels//2 because it concatenates
        row and column embeddings, doubling back to channels.
        This matches official TransAttUNet: self.pos = PositionEmbeddingLearned(512//factor)
        where factor=2, channels=512, so 512//2=256 -> outputs 512.
        """
        super().__init__()
        self.pam           = PAM_Module(channels)
        self.pos           = PositionEmbeddingLearned(channels // 2)
        self.sdpa          = ScaledDotProductAttention(channels)
        self.current_epoch = 0

    def forward(self, x):
        warmup_epochs = 20
        lambd   = max(0.0, min(1.0, self.current_epoch / warmup_epochs))
        x_pam   = self.pam(x)
        x_pos   = x + self.pos(x)
        x_sdpa  = self.sdpa(x_pos)
        out     = x_sdpa + x_pam
        return lambd * out + (1.0 - lambd) * x


# ---------------------------------------------------------------------------
# HTAN_1 — TransAttUNet_R + 1 mHC block at x5
# ---------------------------------------------------------------------------
class HTAN_1(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, expansion_n=4,
                 bilinear=True, hres_only=False, img_size=256):
        super().__init__()
        self.bilinear    = bilinear
        self.expansion_n = expansion_n
        factor           = 2 if bilinear else 1

        # Spatial dims at each level
        self.sp_x5 = img_size // 16   # 256->16
        self.sp_x4 = img_size // 8    # 256->32

        # --- Encoder (identical to official TransAttUNet) ---
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)   # outputs 512ch when bilinear

        # --- mHC at x5 ---
        ch_x5   = 1024 // factor       # 512
        flat_x5 = ch_x5 * self.sp_x5 * self.sp_x5

        self.stream_init = nn.Conv2d(ch_x5, ch_x5 * expansion_n, 1)

        saa_x5        = OfficialSAA(channels=ch_x5)
        reshaping_saa = ReshapingSAA(saa_x5, (ch_x5, self.sp_x5, self.sp_x5))
        self.mhc_x5   = ManifoldConstrainedHyperConnection(
            dim_C=flat_x5,
            expansion_n=expansion_n,
            sub_layer_module=reshaping_saa,
            hres_only=hres_only,
        )
        self.aggregator = nn.Conv2d(ch_x5 * expansion_n, ch_x5, 1)
        self.saa_x5     = saa_x5

        # --- Decoder (identical to official TransAttUNet) ---
        self.up1  = Up(1024, 512 // factor, bilinear)
        self.up2  = Up(1024, 256 // factor, bilinear)
        self.up3  = Up(512,  128 // factor, bilinear)
        self.up4  = Up(256,  64,            bilinear)
        self.outc = OutConv(128, n_classes)

        self.fuse1 = MultiConv(768, 256)
        self.fuse2 = MultiConv(384, 128)
        self.fuse3 = MultiConv(192, 64)
        self.fuse4 = MultiConv(128, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        B, C, H, W = x5.shape

        # mHC at x5
        x5_exp  = self.stream_init(x5)
        x5_str  = x5_exp.view(B, self.expansion_n, -1)
        out_str = self.mhc_x5(x5_str)
        x5      = self.aggregator(out_str.view(B, -1, H, W))

        # Decode — identical to official TransAttUNet
        x6     = self.up1(x5, x4)
        x5_sc  = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_sc, x6), 1)

        x7     = self.up2(x6_cat, x3)
        x6_sc  = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_sc, x7), 1)

        x8     = self.up3(x7_cat, x2)
        x7_sc  = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_sc, x8), 1)

        x9     = self.up4(x8_cat, x1)
        x8_sc  = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9     = torch.cat((x8_sc, x9), 1)

        return self.outc(x9)


# ---------------------------------------------------------------------------
# HTAN_2 — TransAttUNet_R + 2 mHC blocks (x5 + x4)
# ---------------------------------------------------------------------------
class HTAN_2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, expansion_n=4,
                 bilinear=True, hres_only=False, img_size=256):
        super().__init__()
        self.bilinear    = bilinear
        self.expansion_n = expansion_n
        factor           = 2 if bilinear else 1

        self.sp_x5 = img_size // 16
        self.sp_x4 = img_size // 8

        # --- Encoder ---
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # --- mHC at x5 ---
        ch_x5   = 1024 // factor
        flat_x5 = ch_x5 * self.sp_x5 * self.sp_x5

        self.stream_init_x5 = nn.Conv2d(ch_x5, ch_x5 * expansion_n, 1)
        saa_x5              = OfficialSAA(channels=ch_x5)
        self.mhc_x5         = ManifoldConstrainedHyperConnection(
            dim_C=flat_x5, expansion_n=expansion_n,
            sub_layer_module=ReshapingSAA(saa_x5, (ch_x5, self.sp_x5, self.sp_x5)),
            hres_only=hres_only
        )
        self.aggregator_x5 = nn.Conv2d(ch_x5 * expansion_n, ch_x5, 1)
        self.saa_x5        = saa_x5

        # --- mHC at x4 ---
        ch_x4   = 512
        flat_x4 = ch_x4 * self.sp_x4 * self.sp_x4

        self.stream_init_x4 = nn.Conv2d(ch_x4, ch_x4 * expansion_n, 1)
        saa_x4              = OfficialSAA(channels=ch_x4)
        self.mhc_x4         = ManifoldConstrainedHyperConnection(
            dim_C=flat_x4, expansion_n=expansion_n,
            sub_layer_module=ReshapingSAA(saa_x4, (ch_x4, self.sp_x4, self.sp_x4)),
            hres_only=hres_only
        )
        self.aggregator_x4 = nn.Conv2d(ch_x4 * expansion_n, ch_x4, 1)
        self.saa_x4        = saa_x4

        # --- Decoder ---
        self.up1  = Up(1024, 512 // factor, bilinear)
        self.up2  = Up(1024, 256 // factor, bilinear)
        self.up3  = Up(512,  128 // factor, bilinear)
        self.up4  = Up(256,  64,            bilinear)
        self.outc = OutConv(128, n_classes)

        self.fuse1 = MultiConv(768, 256)
        self.fuse2 = MultiConv(384, 128)
        self.fuse3 = MultiConv(192, 64)
        self.fuse4 = MultiConv(128, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        B, C5, H5, W5 = x5.shape
        B, C4, H4, W4 = x4.shape

        # mHC at x5
        x5_exp = self.stream_init_x5(x5)
        x5_str = x5_exp.view(B, self.expansion_n, -1)
        x5     = self.aggregator_x5(self.mhc_x5(x5_str).view(B, -1, H5, W5))

        # mHC at x4
        x4_exp = self.stream_init_x4(x4)
        x4_str = x4_exp.view(B, self.expansion_n, -1)
        x4     = self.aggregator_x4(self.mhc_x4(x4_str).view(B, -1, H4, W4))

        # Decode
        x6     = self.up1(x5, x4)
        x5_sc  = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_sc, x6), 1)

        x7     = self.up2(x6_cat, x3)
        x6_sc  = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_sc, x7), 1)

        x8     = self.up3(x7_cat, x2)
        x7_sc  = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_sc, x8), 1)

        x9     = self.up4(x8_cat, x1)
        x8_sc  = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9     = torch.cat((x8_sc, x9), 1)

        return self.outc(x9)


# ---------------------------------------------------------------------------
# HTAN_1_Hres_only — ablation
# ---------------------------------------------------------------------------
class HTAN_1_Hres_only(HTAN_1):
    def __init__(self, n_channels=3, n_classes=1, expansion_n=4,
                 bilinear=True, img_size=256):
        super().__init__(n_channels=n_channels, n_classes=n_classes,
                         expansion_n=expansion_n, bilinear=bilinear,
                         hres_only=True, img_size=img_size)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
HTAN_MODELS = {
    "htan_1_n2":        lambda: HTAN_1(expansion_n=2),
    "htan_1_n4":        lambda: HTAN_1(expansion_n=4),
    "htan_2_n2":        lambda: HTAN_2(expansion_n=2),
    "htan_2_n4":        lambda: HTAN_2(expansion_n=4),
    "htan_1_hres_only": lambda: HTAN_1_Hres_only(expansion_n=4),
}