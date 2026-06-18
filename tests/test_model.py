import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import HybridEnhancementNet


def test_output_shape():
    model = HybridEnhancementNet()
    x     = torch.randn(2, 3, 256, 256)
    out   = model(x)
    assert out.shape == (2, 3, 256, 256), f"Expected (2,3,256,256), got {out.shape}"
    print("PASS test_output_shape")


def test_output_range():
    """tanh output should be in [-1, 1]"""
    model = HybridEnhancementNet()
    x     = torch.randn(1, 3, 64, 64)
    out   = model(x)
    assert out.min() >= -1.0 and out.max() <= 1.0, "Output outside [-1, 1]"
    print("PASS test_output_range")


def test_variable_input_size():
    """Model should handle any spatial resolution (all convs use padding=1)"""
    model = HybridEnhancementNet()
    for h, w in [(128, 128), (400, 600), (720, 1280)]:
        x   = torch.randn(1, 3, h, w)
        out = model(x)
        assert out.shape == (1, 3, h, w), f"Shape mismatch for {h}x{w}"
    print("PASS test_variable_input_size")


def test_no_nan():
    model = HybridEnhancementNet()
    x     = torch.randn(1, 3, 128, 128)
    out   = model(x)
    assert not torch.isnan(out).any(), "Output contains NaN"
    print("PASS test_no_nan")


if __name__ == '__main__':
    test_output_shape()
    test_output_range()
    test_variable_input_size()
    test_no_nan()
    print("\nAll tests passed.")
