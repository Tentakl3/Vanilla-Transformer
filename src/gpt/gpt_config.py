import torch

class GPTConfig:

    def __init__(self):
        super().__init__()
        self.batch_size = 64 # how many independent sequences will we process in parallel?
        self.block_size = 64 # what is the maximum context length for predictions?
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_embd = 192
        self.n_head = 3
        self.n_layer = 6
        self.dropout = 0.2
        self.vocab_size = self.vocab_load()
        

    def vocab_load(self):
        path = 'src/data/text_data'

        torch.manual_seed(1337)

        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open(f'{path}/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        return len(chars)