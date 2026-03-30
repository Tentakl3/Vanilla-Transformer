import torch
import torch.nn as nn
from torch.nn import functional as F

from .multihead import SA_MultiHeadAttention, FA_MultiHeadAttention
from .feedfoward import FeedFoward, GELU_FeedFoward

dropout = 0.2

class SA_Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = SA_MultiHeadAttention(n_embd, n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FA_Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = FA_MultiHeadAttention(n_embd, n_head, head_size)
        self.ffwd = GELU_FeedFoward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.drop_path = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop_path(self.sa(self.ln1(x)))
        x = x + self.drop_path(self.ffwd(self.ln2(x)))
        #x = x + self.sa(self.ln1(x))
        #x = x + self.ffwd(self.ln2(x))
        return x