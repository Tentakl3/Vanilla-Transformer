import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.1

class PatchEmbedding(nn.Module):
    """ To convert an image into a sequence of patch embeddings """
    def __init__(self, img_size, patch_size, in_channels, n_embd):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_embd, kernel_size=patch_size, stride=patch_size)  # 32 → 8
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            out = self.proj(dummy)
            self.n_patches = out.shape[2] * out.shape[3]

        #self.proj = nn.Conv2d(
        #    in_channels,
        #    n_embd,
        #    kernel_size=patch_size,
        #    stride=patch_size
        #)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, n_embd, H/P, W/P)

        B, C, H, W = x.shape
        x = x.flatten(2)  # (B, n_embd, N)
        x = x.transpose(1, 2)  # (B, N, n_embd)
        return x