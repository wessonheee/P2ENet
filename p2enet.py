"""
P2E-Net: Perception-Preserving Decoupled Enhancement for Real-Time Nighttime Object Detection
Official PyTorch Implementation
Paper: Pattern Analysis and Applications (PAA, Springer Nature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for unsupervised adversarial domain adaptation.
    In the forward pass, it acts as an identity operator.
    In the backward pass, it negates gradients by multiplying with -lambda_val.
    """
    @staticmethod
    def forward(ctx, x, lambda_val=1.0):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def grad_reverse(x, lambda_val=1.0):
    return GradientReversalFunction.apply(x, lambda_val)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention module.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ca(x)


class GatedBlock(nn.Module):
    """
    Core building block of P2E-Net:
    1. Depthwise separable spatial convolution with 2x channel expansion
    2. Non-linear gating (SimpleGate: x1 * sigmoid(x2))
    3. Pointwise projection
    4. Gated Feed-Forward Network (FFN)
    5. Channel attention + Residual skip connection
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        # Depthwise Conv 3x3 with 2x channel expansion
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, groups=dim)
        # Pointwise Conv 1x1
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        # Gated FFN (2x Expand)
        self.conv3 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1)
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        # 1. Depthwise spatial filtering + non-linear gating
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.conv2(x)
        
        # 2. Gated Feed-Forward Network (FFN)
        f1, f2 = self.conv3(x).chunk(2, dim=1)
        x = f1 * torch.sigmoid(f2)
        x = self.conv4(x)
        
        # 3. Channel attention
        x = x * self.ca(x)
        return res + x


class P2ENet(nn.Module):
    """
    P2E-Net (Perception-Preserving Enhancement Network)
    - Ultra-lightweight U-Net topology: 0.701M (~0.71M) parameters, 8.9 GFLOPs.
    - Bilateral Structure-Variance Modulation (BSVM) bottleneck.
    - Long-range global identity skip: I_enh = clip(I_in + Delta I, 0, 1).
    """
    def __init__(self, base_c: int = 36):
        super().__init__()
        self.base_c = base_c
        
        # Initial projection: RGB (3 channels) -> base_c (36 channels)
        self.in_conv = nn.Conv2d(3, base_c, kernel_size=3, padding=1)
        
        # Encoder Stage 1 (C=36) & Downsample 1
        self.enc1 = GatedBlock(base_c)
        self.down1 = nn.Conv2d(base_c, base_c * 2, kernel_size=2, stride=2)
        
        # Encoder Stage 2 (2C=72) & Downsample 2
        self.enc2 = GatedBlock(base_c * 2)
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, kernel_size=2, stride=2)
        
        # Encoder Stage 3 (4C=144)
        self.enc3 = GatedBlock(base_c * 4)
        
        # Dual Bottleneck (BSVM: Semantic Stream & Visual Stream)
        self.bottleneck_sem = GatedBlock(base_c * 4)
        self.bottleneck_vis = GatedBlock(base_c * 4)
        
        # Bottleneck Affine Modulation: produces [gamma, beta] maps
        self.mod_conv = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c * 4, base_c * 8, kernel_size=1)
        )
        
        # Decoder Stage 2 (Upsample 2 -> 2C=72)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = GatedBlock(base_c * 2)
        
        # Decoder Stage 1 (Upsample 1 -> C=36)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = GatedBlock(base_c)
        
        # Output Reconstruction Conv
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic shape alignment for arbitrary input dimensions.
        """
        x_in = x
        x = self.in_conv(x)
        
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        e3 = self.enc3(d2)
        
        # BSVM Dual Bottleneck Streams
        f_sem = self.bottleneck_sem(e3)
        f_vis = self.bottleneck_vis(e3)
        
        # Affine Modulation: F_mod = F_vis * (1 + gamma) + beta
        mod_params = self.mod_conv(f_sem)
        gamma, beta = mod_params.chunk(2, dim=1)
        f_mod = f_vis * (1.0 + gamma) + beta
        
        # Decoder Stage 2
        u2 = self.up2(f_mod)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        u2 = u2 + e2
        d_out2 = self.dec2(u2)
        
        # Decoder Stage 1
        u1 = self.up1(d_out2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        u1 = u1 + e1
        d_out1 = self.dec1(u1)
        
        # Global identity residual addition
        delta = self.out_conv(d_out1)
        out = torch.clamp(x_in + delta, 0.0, 1.0)
        return out


class GatedFusionModule(nn.Module):
    """
    Illumination-Gated Fusion Mechanism (Section 3.6):
    Dynamically blends raw input with enhanced output based on local luminance:
      I_final = G_alpha(I_in) * I_in + (1 - G_alpha(I_in)) * I_enh
    where:
      G_alpha(I_in) = clip((I_gray - tau_dark) / (tau_bright - tau_dark), 0, 1)^alpha
      
    Operating Modes:
      - High-Recall Mode: alpha = 0.0 (maximizing target visibility in dark scenes)
      - Balanced Mode:    alpha = 0.5
      - High-Precision:   alpha = 0.8 (maximizing noise immunity and edge stability)
    """
    def __init__(self, tau_dark: float = 0.05, tau_bright: float = 0.40):
        super().__init__()
        self.tau_dark = tau_dark
        self.tau_bright = tau_bright

    def forward(self, img_in: torch.Tensor, img_enh: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        if alpha <= 0.0:
            return img_enh
        
        # Compute grayscale luminance: 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * img_in[:, 0:1, :, :] + 0.587 * img_in[:, 1:2, :, :] + 0.114 * img_in[:, 2:3, :, :]
        norm_illum = torch.clamp((gray - self.tau_dark) / (self.tau_bright - self.tau_dark + 1e-6), 0.0, 1.0)
        gate = torch.pow(norm_illum, alpha)
        
        fused = gate * img_in + (1.0 - gate) * img_enh
        return torch.clamp(fused, 0.0, 1.0)


class DomainDiscriminator(nn.Module):
    """
    Adversarial Domain Discriminator for Backbone Feature Adaptation (Section 3.5).
    Distinguishes daytime clean feature distributions from nighttime enhanced features.
    Integrated with Gradient Reversal Layer (GRL).
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
        x_rev = grad_reverse(x, lambda_grl)
        return self.net(x_rev)


def build_p2enet(checkpoint_path: Optional[str] = None, device: str = 'cpu') -> P2ENet:
    """
    Factory helper to initialize P2ENet and optionally load weights.
    """
    model = P2ENet(base_c=36)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"[P2ENet] Successfully loaded checkpoint from: {checkpoint_path}")
    model = model.to(device)
    return model
