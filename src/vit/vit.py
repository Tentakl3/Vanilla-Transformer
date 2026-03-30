import torch
import torch.nn as nn
from torch.nn import functional as F

from .patch_embd import PatchEmbedding
from ..transformer.block import FA_Block

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.1

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels,
                 n_embd, n_head, n_layer, num_classes):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, n_embd)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, n_embd))

        self.blocks = nn.Sequential(*[
            FA_Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, N, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding

        x = self.blocks(x)
        x = self.ln_f(x)

        cls_output = x[:, 0]  # take CLS token
        logits = self.head(cls_output)

        return logits