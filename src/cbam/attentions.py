import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, channels : int, reduction : int = 16):

        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x : torch.Tensor):

        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = torch.sigmoid(avg + mx)

        return x

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size : int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=False
        )

    def forward(self, x: torch.Tensor):

        avg = x.mean(dim=1, keepdim=True)
        max = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, max], dim=1)))



