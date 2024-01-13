import torch

batch_size = 2
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 3200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

vocab_size = 45 
sos_token = vocab_size - 3 
eos_token = vocab_size - 2
pad_token = vocab_size - 1