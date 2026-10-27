"""
P2E-Net Downstream Nighttime Object Detection Benchmark
Integrates P2E-Net as a frozen zero-shot visual enhancement front-end
feeding directly into standard YOLOv8s on the ExDark benchmark.
"""

import os
import argparse
import time
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from models.p2enet import P2ENet, GatedFusionModule, build_p2enet


def parse_args():
    parser = argparse.ArgumentParser(description="P2E-Net Downstream Detection Evaluation")
    parser.add_argument("--img_dir", type=str, default="data/samples",
                        help="Path to ExDark evaluation images.")
    parser.add_argument("--p2enet_checkpoint", type=str, default="checkpoints/p2enet_exdark_adapted.pth",
                        help="Path to adapted P2E-Net checkpoint.")
    parser.add_argument("--yolo_weights", type=str, default="yolov8s.pt",
                        help="Path to pretrained YOLOv8s checkpoint.")
    parser.add_argument("--mode", type=str, default="high_recall",
                        choices=["raw", "high_recall", "high_precision"],
                        help="Evaluation mode: 'raw' (no enhancement), 'high_recall' (alpha=0.0), 'high_precision' (alpha=0.8).")
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                        help="Confidence threshold for bounding box predictions.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("P2E-Net Downstream Object Detection Evaluation Pipeline")
    print(f"Device:            {args.device}")
    print(f"Evaluation Mode:   {args.mode}")
    print(f"P2E-Net Weights:   {args.p2enet_checkpoint}")
    print(f"YOLOv8 Weights:    {args.yolo_weights}")
    print("=" * 65)

    # 1. Load YOLOv8 Detector
    try:
        from ultralytics import YOLO
        detector = YOLO(args.yolo_weights)
        print(f"[Detector] Loaded YOLOv8s weights: {args.yolo_weights}")
    except ImportError:
        print("[Notice] 'ultralytics' library not detected.")
        print("To run live YOLO detection, install: pip install ultralytics")
        detector = None

    # 2. Load P2E-Net Front-end (if not raw mode)
    p2enet = None
    fusion = None
    if args.mode != "raw":
        if os.path.exists(args.p2enet_checkpoint):
            p2enet = build_p2enet(checkpoint_path=args.p2enet_checkpoint, device=args.device)
            p2enet.eval()
            fusion = GatedFusionModule().to(args.device)
            print(f"[P2E-Net] Front-end initialized with mode '{args.mode}'.")
        else:
            print(f"[Warning] P2E-Net checkpoint '{args.p2enet_checkpoint}' not found.")

    # 3. Benchmark Sample Predictions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    if os.path.exists(args.img_dir):
        files = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith(valid_exts)]
        print(f"Found {len(files)} test images in: {args.img_dir}")
        
        for p in files[:3]:
            raw_img = Image.open(p).convert("RGB")
            
            # Preprocess through P2E-Net
            if p2enet is not None:
                t_in = TF.to_tensor(raw_img).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    t_enh = p2enet(t_in)
                    alpha = 0.8 if args.mode == "high_precision" else 0.0
                    if alpha > 0.0:
                        t_enh = fusion(t_in, t_enh, alpha=alpha)
                feed_img = TF.to_pil_image(t_enh.squeeze(0).cpu())
            else:
                feed_img = raw_img

            if detector is not None:
                results = detector.predict(source=feed_img, conf=args.conf_thresh, verbose=False)
                num_boxes = len(results[0].boxes) if results else 0
                print(f"  Image: {os.path.basename(p):30s} -> Detected {num_boxes} objects")
            else:
                print(f"  Image: {os.path.basename(p):30s} -> Preprocessed successfully (YOLO not installed)")
    else:
        print(f"Image directory '{args.img_dir}' does not exist.")

    print("=" * 65)
    print("Benchmark complete. Refer to manuscript Table 2 for full ExDark 12-class mAP.")
    print("=" * 65)


if __name__ == "__main__":
    main()
