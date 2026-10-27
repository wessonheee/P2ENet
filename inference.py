"""
P2E-Net Inference & Visual Enhancement Script
Usage:
  python inference.py --input data/samples/exdark_sample_motorcycle.jpg --output results/
  python inference.py --input data/samples/ --output results/ --mode high_precision
"""

import os
import argparse
import time
from PIL import Image
import torch
import torchvision.transforms.functional as TF
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"Starting {desc}...")
        return iterable

from models.p2enet import P2ENet, GatedFusionModule, build_p2enet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="P2E-Net Inference & Image Enhancement")
    parser.add_argument("--input", type=str, default=os.path.join(SCRIPT_DIR, "data", "samples"),
                        help="Path to an input image or directory of images.")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "results"),
                        help="Directory to save enhanced output images.")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SCRIPT_DIR, "checkpoints", "p2enet_sota_best.pth"),
                        help="Path to pretrained P2E-Net weights (.pth).")
    parser.add_argument("--mode", type=str, default="high_recall",
                        choices=["enhancement", "high_recall", "high_precision", "custom"],
                        help="Operating mode: 'enhancement' (pure), 'high_recall' (alpha=0.0), 'high_precision' (alpha=0.8).")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Custom alpha for Gated Fusion [0.0, 1.0]. Overrides --mode.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device ('cuda' or 'cpu').")
    return parser.parse_args()


def get_alpha(mode: str, custom_alpha: float = None) -> float:
    if custom_alpha is not None:
        return custom_alpha
    if mode == "enhancement" or mode == "high_recall":
        return 0.0
    elif mode == "high_precision":
        return 0.8
    return 0.0


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    alpha = get_alpha(args.mode, args.alpha)
    print("=" * 60)
    print("P2E-Net Inference Pipeline")
    print(f"Device:           {args.device}")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Mode:             {args.mode} (alpha={alpha:.2f})")
    print(f"Input:            {args.input}")
    print(f"Output Directory: {args.output}")
    print("=" * 60)

    # 1. Initialize Model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Please download or provide valid path.")
    
    model = build_p2enet(checkpoint_path=args.checkpoint, device=args.device)
    model.eval()
    fusion_module = GatedFusionModule().to(args.device)

    # 2. Gather Image Paths
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')
    if os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith(valid_exts)
        ]
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    if len(image_paths) == 0:
        print(f"No valid images found in {args.input}.")
        return

    # 3. Warm-up GPU (only if sufficient VRAM >= 4GB)
    if args.device == "cuda":
        torch.cuda.empty_cache()
        try:
            free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if free_mem >= 4.0:
                dummy = torch.randn(1, 3, 640, 640, device=args.device)
                for _ in range(3):
                    _ = model(dummy)
                torch.cuda.synchronize()
        except Exception:
            pass

    # 4. Process Images
    total_time = 0.0
    for img_path in tqdm(image_paths, desc="Enhancing Images"):
        raw_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_pil.size
        
        # Convert to tensor [1, 3, H, W]
        t_in = TF.to_tensor(raw_pil).unsqueeze(0).to(args.device)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        try:
            with torch.no_grad():
                t_enh = model(t_in)
                if alpha > 0.0:
                    t_final = fusion_module(t_in, t_enh, alpha=alpha)
                else:
                    t_final = t_enh
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\n[Warning] CUDA out of memory detected. Falling back to CPU for processing...")
                torch.cuda.empty_cache()
                model_cpu = model.to("cpu")
                fusion_cpu = fusion_module.to("cpu")
                with torch.no_grad():
                    t_enh = model_cpu(t_in.cpu())
                    if alpha > 0.0:
                        t_final = fusion_cpu(t_in.cpu(), t_enh, alpha=alpha)
                    else:
                        t_final = t_enh
            else:
                raise e

        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        total_time += (t1 - t0)

        # Save enhanced image
        out_pil = TF.to_pil_image(t_final.squeeze(0).cpu())
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, f"{base_name}_p2enet.jpg")
        out_pil.save(out_path, quality=95)

    avg_latency = (total_time / len(image_paths)) * 1000.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    print("-" * 60)
    print(f"Processed {len(image_paths)} images successfully.")
    print(f"Average Forward Latency: {avg_latency:.2f} ms ({avg_fps:.1f} FPS)")
    print(f"Enhanced results saved to: {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
