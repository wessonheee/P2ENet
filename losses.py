"""
P2E-Net Loss Functions Formulation
Includes:
1. Highlight Protection and Glare Immunity Loss (L_hp, Section 3.3)
2. Perceptual Structural Variance Loss (L_pv, Section 3.4)
3. Adversarial Domain Adaptation Loss (L_adapt, Section 3.5)
4. Total Compound Synergistic Objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional


class HighlightProtectionLoss(nn.Module):
    """
    Highlight Protection and Glare Immunity Loss (L_hp):
    Asymmetric barrier penalty preventing non-linear saturation at detector inputs.
    Strictly penalizes luminance overshoot (I_enh > I_in) in pre-existing bright zones
    (streetlights, headlights, signage) above threshold tau_high.
    
    Formula:
      M_high = sigmoid(kappa * (I_in_gray - tau_high))
      L_hp = (1 / HW) * sum(M_high * ReLU(I_enh - I_in))
    """
    def __init__(self, tau_high: float = 0.30, kappa: float = 20.0):
        super().__init__()
        self.tau_high = tau_high
        self.kappa = kappa

    def forward(self, img_in: torch.Tensor, img_enh: torch.Tensor) -> torch.Tensor:
        # Input luminance channel: 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * img_in[:, 0:1, :, :] + 0.587 * img_in[:, 1:2, :, :] + 0.114 * img_in[:, 2:3, :, :]
        mask_high = torch.sigmoid(self.kappa * (gray - self.tau_high))
        
        # Asymmetric penalty: only positive overshoot I_enh > I_in is penalized
        overshoot = F.relu(img_enh - img_in)
        loss = torch.mean(mask_high * overshoot)
        return loss


class SobelOperator(nn.Module):
    """
    Differentiable 2D Sobel gradient operator for edge & structural variance extraction.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        # Tile across in_channels
        self.weight_x = nn.Parameter(sobel_x.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.groups = in_channels

    def forward(self, x: torch.Tensor) -> Tuple_Gradient:
        gx = F.conv2d(x, self.weight_x, padding=1, groups=self.groups)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=self.groups)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


class TotalVariationLoss(nn.Module):
    """
    Total Variation regularization to suppress high-frequency noise amplification in dark regions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(dh) + torch.mean(dw)


class PerceptualVarianceLoss(nn.Module):
    """
    Perceptual Structural Variance Loss (L_pv):
    Extracts multi-scale deep features from pre-trained VGG-19 (layers relu1_2, relu2_2, relu3_2),
    computes structural Sobel gradients on feature maps, and penalizes structural distortion
    while regularizing sensor noise via Total Variation.
    
    Formula:
      L_pv = sum_l (1 / H_l W_l) * ||nabla phi_l(I_enh) - nabla phi_l(I_gt)||_1 + lambda_tv * L_TV(I_enh)
    """
    def __init__(self, lambda_tv: float = 1e-4, device: str = 'cpu'):
        super().__init__()
        self.lambda_tv = lambda_tv
        self.tv_loss = TotalVariationLoss()
        
        # Load pre-trained VGG-19 feature slices
        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)]).to(device)   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)]).to(device) # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 14)]).to(device) # relu3_2
        
        self.sobel1 = SobelOperator(64).to(device)
        self.sobel2 = SobelOperator(128).to(device)
        self.sobel3 = SobelOperator(256).to(device)
        
        # VGG ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, img_enh: torch.Tensor, img_gt: torch.Tensor) -> torch.Tensor:
        enh_norm = self._normalize(img_enh)
        gt_norm = self._normalize(img_gt)
        
        # Layer 1
        h1_enh = self.slice1(enh_norm)
        h1_gt = self.slice1(gt_norm)
        grad1_enh = self.sobel1(h1_enh)
        grad1_gt = self.sobel1(h1_gt)
        l1 = F.l1_loss(grad1_enh, grad1_gt)
        
        # Layer 2
        h2_enh = self.slice2(h1_enh)
        h2_gt = self.slice2(h1_gt)
        grad2_enh = self.sobel2(h2_enh)
        grad2_gt = self.sobel2(h2_gt)
        l2 = F.l1_loss(grad2_enh, grad2_gt)
        
        # Layer 3
        h3_enh = self.slice3(h2_enh)
        h3_gt = self.slice3(h2_gt)
        grad3_enh = self.sobel3(h3_enh)
        grad3_gt = self.sobel3(h3_gt)
        l3 = F.l1_loss(grad3_enh, grad3_gt)
        
        tv = self.tv_loss(img_enh)
        return (l1 + l2 + l3) / 3.0 + self.lambda_tv * tv


class P2ENetCompoundLoss(nn.Module):
    """
    Comprehensive Synergistic Loss Objective (Equation 13):
      L_total = L1(I_enh, I_gt) + lambda_hp * L_hp + lambda_pv * L_pv + lambda_adapt * L_adapt
    where default coefficients from paper:
      lambda_hp = 2.0
      lambda_pv = 0.10
      lambda_adapt = 0.05
    """
    def __init__(
        self,
        lambda_hp: float = 2.0,
        lambda_pv: float = 0.10,
        lambda_adapt: float = 0.05,
        device: str = 'cpu'
    ):
        super().__init__()
        self.lambda_hp = lambda_hp
        self.lambda_pv = lambda_pv
        self.lambda_adapt = lambda_adapt
        
        self.l1_loss = nn.L1Loss()
        self.hp_loss = HighlightProtectionLoss(tau_high=0.30, kappa=20.0)
        self.pv_loss = PerceptualVarianceLoss(device=device)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        img_in: torch.Tensor,
        img_enh: torch.Tensor,
        img_gt: Optional[torch.Tensor] = None,
        disc_pred_fake: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=img_in.device)
        
        # 1. Photometric Visual Reconstruction Loss (L1)
        if img_gt is not None:
            l1 = self.l1_loss(img_enh, img_gt)
            losses['l1'] = l1
            total = total + l1
            
            # 2. Perceptual Structural Variance Loss (L_pv)
            pv = self.pv_loss(img_enh, img_gt)
            losses['l_pv'] = pv
            total = total + self.lambda_pv * pv

        # 3. Highlight Protection Loss (L_hp)
        hp = self.hp_loss(img_in, img_enh)
        losses['l_hp'] = hp
        total = total + self.lambda_hp * hp

        # 4. Adversarial Perceptual Domain Adaptation Loss (L_adapt)
        if disc_pred_fake is not None:
            # Generator wants discriminator to classify nighttime enhanced feature as daytime (target = 1)
            target_real = torch.ones_like(disc_pred_fake)
            adapt = self.bce_loss(disc_pred_fake, target_real)
            losses['l_adapt'] = adapt
            total = total + self.lambda_adapt * adapt

        losses['total'] = total
        return losses
