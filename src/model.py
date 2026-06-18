import torch
import torch.nn as nn


class HybridEnhancementNet(nn.Module):
    """
    Hybrid low-light image enhancement network combining:
    - Zero-DCE style sequential convolutions (conv1-conv4)
    - CIDNet style skip connections with concatenation (conv5-conv7)

    Architecture:
        Input (3, H, W)
            |
        conv1 -> conv2 -> conv3 -> conv4
                            |          |
                      [cat(conv4, conv3)] -> conv5
                                  |
                      [cat(conv5, conv2)] -> conv6
                                  |
                      [cat(conv6, conv1)] -> conv7
                                  |
                          output_layer (tanh)
                                  |
                         Output (3, H, W)

    All conv layers: 3x3 kernel, padding=1 (preserves spatial dims)
    Weights: Kaiming normal initialization
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Zero-DCE style sequential layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # CIDNet style concatenation layers
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)   # cat(conv4, conv3)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)   # cat(conv5, conv2)
        self.conv7 = nn.Conv2d(64, 24, kernel_size=3, padding=1)   # cat(conv6, conv1)

        # Final output layer
        self.output_layer = nn.Conv2d(24, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Sequential feature extraction (Zero-DCE style)
        c1 = torch.relu(self.conv1(x))    # [B, 32, H, W]
        c2 = torch.relu(self.conv2(c1))   # [B, 32, H, W]
        c3 = torch.relu(self.conv3(c2))   # [B, 32, H, W]
        c4 = torch.relu(self.conv4(c3))   # [B, 32, H, W]

        # Skip connections with concatenation (CIDNet style)
        i1 = torch.cat([c4, c3], dim=1)   # [B, 64, H, W]
        c5 = torch.relu(self.conv5(i1))   # [B, 32, H, W]

        i2 = torch.cat([c5, c2], dim=1)   # [B, 64, H, W]
        c6 = torch.relu(self.conv6(i2))   # [B, 32, H, W]

        i3 = torch.cat([c6, c1], dim=1)   # [B, 64, H, W]
        c7 = torch.relu(self.conv7(i3))   # [B, 24, H, W]

        # Final enhancement with tanh to keep output in [-1, 1]
        out = torch.tanh(self.output_layer(c7))  # [B, 3, H, W]
        return out
