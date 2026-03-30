import torch
import torch.nn as nn
from torch.nn import functional as F

from .head import SA_Head, FA_Head
from ..gpt.gpt_config import GPTConfig

dropout = 0.2
config = GPTConfig()

class SA_MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, 
        n_embd: int, 
        num_heads: int, 
        head_size: int
    ):
        
        super().__init__()
        self.heads = nn.ModuleList([SA_Head(n_embd, head_size, block_size=config.block_size) for _ in range(num_heads)])

        """ the projection operator works as a weighted sum from the attention head outputs """
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        """ concatenation of attention outputs as a single matrix """
        out = self.dropout(self.proj(out))
        return out
    

class FA_MultiHeadAttention(nn.Module):
    """ multiple heads of full-attention in parallel """

    def __init__(self, 
        n_embd: int, 
        num_heads: int, 
        head_size: int
    ):
        
        super().__init__()
        self.heads = nn.ModuleList([FA_Head(n_embd, head_size) for _ in range(num_heads)])

        """ the projection operator works as a weighted sum from the attention head outputs """
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        """ concatenation of attention outputs as a single matrix """
        out = self.dropout(self.proj(out))
        return out