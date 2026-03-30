import torch
import torch.nn as nn
import tiktoken
from torch.nn import functional as F

class IntTokens:
    def __init__(self):
        super().__init__()
        self.path = 'src/data/text_data'
        self.chars, self.text = self.read_path()
    
    def read_path(self):
        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open(f'{self.path}/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        return chars, text
    
    def encode(self):
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        encode = lambda s: [stoi[c] for c in s]
        return encode

    def decode(self):
        itos = { i:ch for i,ch in enumerate(self.chars) }
        decode = lambda l: ''.join([itos[i] for i in l])
        return decode

class TikTokens:
    def __init__(self):
        super().__init__()
        self.path = 'src/data/text_data'
        self.encoding = tiktoken.get_encoding("gpt2")
        self.chars, self.text = self.read_path()
    
    def read_path(self):
        with open(f'{self.path}/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(text)

        return chars, text

    def encode(self):
        encode = lambda s: [self.encoding.encode(c) for c in s]
        return encode

    def decode(self):
        decode = lambda l: ''.join([self.encoding.decode_single_token_bytes(i) for i in l])
        return decode