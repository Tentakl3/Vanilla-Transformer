import torch
import torch.nn as nn
from torch.nn import functional as F

from .gpt import GPTLanguageModel
from .encode import IntTokens, TikTokens
from .gpt_config import GPTConfig

config = GPTConfig()

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
# ------------

tokenizer = IntTokens()
chars, text = tokenizer.chars, tokenizer.text

encode = tokenizer.encode()
decode = tokenizer.decode()

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading

def get_batch_sequential(data, start_idx):
    x = data[start_idx:start_idx + config.block_size]
    y = data[start_idx + 1:start_idx + config.block_size + 1]
    return x.unsqueeze(0), y.unsqueeze(0)

def get_batch_2(split, step):
    data = train_data if split == 'train' else val_data
    
    start_positions = torch.arange(
        step * config.batch_size * config.block_size,
        (step + 1) * config.batch_size * config.block_size,
        config.block_size
    )
    
    x = torch.stack([data[i:i+config.block_size] for i in start_positions])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in start_positions])
    
    return x.to(config.device), y.to(config.device)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    if len(x.shape) > 2:
        x = x.squeeze(-1)
    if len(y.shape) > 2:
        y = y.squeeze(-1)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':
    model = GPTLanguageModel()
    m = model.to(config.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
