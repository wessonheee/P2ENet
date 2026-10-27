"""
P2E-Net Training Pipeline
Supports 3 progressive training stages:
  - Stage 1: Photometric Visual Pre-training on Paired Data (LOL-v1) with L1 + L_hp
  - Stage 2: Perceptual Structural Variance Refinement with L1 + L_hp + L_pv
  - Stage 3: Unsupervised Adversarial Domain Adaptation on Target Nighttime Data (ExDark) with L_adapt
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models.p2enet import P2ENet, DomainDiscriminator, build_p2enet
from losses.losses import P2ENetCompoundLoss, HighlightProtectionLoss, PerceptualVarianceLoss
from data.dataset import PairedLowLightDataset, UnpairedLowLightDataset


def parse_args():
    parser = argparse.ArgumentParser(description="P2E-Net Training Pipeline")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Training stage: 1 (Photometric), 2 (Perceptual), 3 (Domain Adaptation).")
    parser.add_argument("--data_dir", type=str, default="data/lol_v1",
                        help="Path to paired training dataset directory.")
    parser.add_argument("--target_dir", type=str, default="data/exdark",
                        help="Path to target domain unannotated nighttime images (Stage 3).")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume or fine-tune from.")
    parser.add_argument("--save_dir", type=str, default="checkpoints/experiments", help="Directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on ('cuda' or 'cpu').")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP).")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    stage: int,
    scaler: torch.cuda.amp.GradScaler = None,
    discriminator: nn.Module = None,
    opt_disc: optim.Optimizer = None
) -> float:
    model.train()
    if discriminator is not None:
        discriminator.train()
        
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        low_imgs, high_imgs, _ = batch
        low_imgs = low_imgs.to(device)
        high_imgs = high_imgs.to(device)
        
        optimizer.zero_grad()
        if opt_disc is not None:
            opt_disc.zero_grad()

        # Mixed precision forward
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            enh_imgs = model(low_imgs)
            
            disc_pred_fake = None
            if stage == 3 and discriminator is not None:
                # Discriminate between enhanced nighttime and clean reference
                disc_pred_fake = discriminator(enh_imgs)

            loss_dict = criterion(
                img_in=low_imgs,
                img_enh=enh_imgs,
                img_gt=high_imgs,
                disc_pred_fake=disc_pred_fake
            )
            loss = loss_dict['total']

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if opt_disc is not None:
                scaler.step(opt_disc)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            if opt_disc is not None:
                opt_disc.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 65)
    print(f"P2E-Net Training Pipeline - Stage {args.stage}")
    print(f"Device:           {args.device}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Epochs:           {args.epochs}")
    print(f"Initial LR:       {args.lr}")
    print(f"Save Directory:   {args.save_dir}")
    print("=" * 65)

    # 1. Initialize Network
    model = P2ENet(base_c=36).to(args.device)
    if args.resume and os.path.exists(args.resume):
        state_dict = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[P2ENet] Resumed weights from: {args.resume}")

    discriminator = None
    opt_disc = None
    if args.stage == 3:
        discriminator = DomainDiscriminator(in_channels=3).to(args.device)
        opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

    # 2. Setup Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and args.device == "cuda") else None

    # 3. Setup Loss Objective
    if args.stage == 1:
        # Stage 1: Photometric L1 + L_hp
        criterion = P2ENetCompoundLoss(lambda_hp=2.0, lambda_pv=0.0, lambda_adapt=0.0, device=args.device)
    elif args.stage == 2:
        # Stage 2: Add Structural Variance L_pv
        criterion = P2ENetCompoundLoss(lambda_hp=2.0, lambda_pv=0.10, lambda_adapt=0.0, device=args.device)
    else:
        # Stage 3: Full Orthogonal Quad-Domain Synergistic Framework
        criterion = P2ENetCompoundLoss(lambda_hp=2.0, lambda_pv=0.10, lambda_adapt=0.05, device=args.device)

    # 4. Setup Data Loader
    if os.path.exists(args.data_dir):
        dataset = PairedLowLightDataset(root_dir=args.data_dir, is_train=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        print(f"Loaded {len(dataset)} training pairs from: {args.data_dir}")
    else:
        print(f"[Notice] Dataset directory '{args.data_dir}' not found.")
        print("To start actual training, download the LOL-v1 dataset to data/lol_v1.")
        return

    # 5. Training Loop
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=args.device,
            stage=args.stage,
            scaler=scaler,
            discriminator=discriminator,
            opt_disc=opt_disc
        )
        scheduler.step()
        epoch_time = time.time() - t0

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.1f}s")

        # Save checkpoint periodically
        if avg_loss < best_loss or epoch % 10 == 0:
            best_loss = min(best_loss, avg_loss)
            ckpt_name = f"p2enet_stage{args.stage}_epoch_{epoch:03d}.pth"
            save_path = os.path.join(args.save_dir, ckpt_name)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved checkpoint to: {save_path}")

    print("=" * 65)
    print("Training Completed Successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
