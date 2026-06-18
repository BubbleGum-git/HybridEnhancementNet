"""
Export trained HybridEnhancementNet to ONNX format for rover deployment.

Usage:
    python deploy/export_onnx.py \
        --checkpoint /path/to/best_model.pth \
        --output models/exported/model.onnx \
        --input-size 256 256
"""

import argparse
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import HybridEnhancementNet


def export_onnx(checkpoint_path, output_path, input_h=256, input_w=256):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device('cpu')  # export on CPU for compatibility
    model  = HybridEnhancementNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, input_h, input_w)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    )

    print(f"Exported ONNX model to {output_path}")

    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', default='models/exported/model.onnx')
    parser.add_argument('--input-size', nargs=2, type=int, default=[256, 256])
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.input_size[0], args.input_size[1])
