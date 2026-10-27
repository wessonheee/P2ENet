"""
P2E-Net Benchmark Script
Measures:
1. Total & Trainable Parameter Count
2. Theoretical FLOPs / MACs at user-specified resolutions
3. Real forward latency (ms) and throughput (FPS)
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from models.p2enet import P2ENet, build_p2enet


def calculate_macs(model: nn.Module, input_size=(1, 3, 640, 640)) -> int:
    """
    Hook-based MACs (Multiply-Accumulate operations) calculator for Conv2d and ConvTranspose2d.
    """
    total_macs = 0
    hooks = []

    def conv_hook(module, inp, output):
        nonlocal total_macs
        batch_size, out_channels, out_h, out_w = output.shape
        in_channels = module.in_channels
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (in_channels // module.groups)
        macs = batch_size * out_channels * out_h * out_w * kernel_ops
        total_macs += macs

    def conv_transpose_hook(module, inp, output):
        nonlocal total_macs
        batch_size, out_channels, out_h, out_w = output.shape
        in_channels = module.in_channels
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (in_channels // module.groups)
        macs = batch_size * out_channels * out_h * out_w * kernel_ops
        total_macs += macs

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.ConvTranspose2d):
            hooks.append(m.register_forward_hook(conv_transpose_hook))

    inp = torch.randn(*input_size, device=next(model.parameters()).device)
    with torch.no_grad():
        _ = model(inp)

    for h in hooks:
        h.remove()

    return total_macs


def parse_args():
    parser = argparse.ArgumentParser(description="P2E-Net Latency & Complexity Benchmark")
    parser.add_argument("--height", type=int, default=640, help="Benchmark image height.")
    parser.add_argument("--width", type=int, default=640, help="Benchmark image width.")
    parser.add_argument("--repeats", type=int, default=50, help="Number of benchmark timing iterations.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to benchmark ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("P2E-Net Architectural Complexity & Efficiency Benchmark")
    print(f"Device:            {args.device}")
    print(f"Test Resolution:   {args.height} x {args.width}")
    print(f"Timing Iterations: {args.repeats}")
    print("=" * 65)

    # 1. Initialize Model
    model = P2ENet(base_c=36).to(args.device)
    model.eval()

    # 2. Parameter Counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:      {total_params:,} ({total_params / 1e6:.4f} M)")
    print(f"Trainable Parameters:  {trainable_params:,} ({trainable_params / 1e6:.4f} M)")

    # 3. MACs / FLOPs Computation
    input_shape = (1, 3, args.height, args.width)
    macs = calculate_macs(model, input_shape)
    gflops = macs / 1e9
    print(f"Computational MACs:    {macs:,} ({gflops:.2f} GFLOPs)")

    # 4. Latency Benchmarking
    inp = torch.randn(*input_shape, device=args.device)

    # Warmup
    warmup_count = 5 if args.device == "cuda" else 2
    with torch.no_grad():
        for _ in range(warmup_count):
            _ = model(inp)
        if args.device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(args.repeats):
            _ = model(inp)
            if args.device == "cuda":
                torch.cuda.synchronize()
        t1 = time.time()

    avg_latency = (t1 - t0) / args.repeats * 1000.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("-" * 65)
    print(f"Average Forward Time:  {avg_latency:.2f} ms")
    print(f"Inference Throughput:  {fps:.1f} FPS")
    print("=" * 65)


if __name__ == "__main__":
    main()
