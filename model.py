import torch
import torch.nn as nn
from torch.nn import functional as F



# Define a class for the self-attention head
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask=None):
        B, T, C = memory.shape
        k = self.key(memory)
        q = self.query(x)
        v = self.value(memory)

        wei = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5))
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v

        return out

# Define a class for multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Define a class for the feed-forward layer
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Define a class for the transformer block
class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

# Define a class for cross-attention
class CrossAttention(nn.Module):
    def __init__(self, n_embd, head_size, n_head, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask=None):
        # Apply each head to get a list of attention outputs
        attn_outputs = [head(x, memory, mask) for head in self.heads]
        # Concatenate all attention outputs
        concat_output = torch.cat(attn_outputs, dim=-1)
        # Project back to the original embedding dimension
        projected_output = self.proj(concat_output)
        return projected_output


# Define a class for the encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

# Define a class for the decoder block
class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attn = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.cross_attn = CrossAttention(n_embd, head_size, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, encoder_output):
        B, T, _ = x.shape
        self_attn_mask = self.tril[:T, :T]
        self_attn_mask = self_attn_mask.unsqueeze(0)
        self_attn_mask = self_attn_mask.expand(B, -1, -1)
        x = x + self.self_attn(self.ln1(x), self_attn_mask)
        x = x + self.cross_attn(self.ln2(x), encoder_output)
        x = x + self.ffwd(self.ln3(x))
        return x


# Define a class for the decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, encoder_output, targets=None, target_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, encoder_output)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            if target_mask is not None:
                mask_flat = target_mask.view(B * T)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
            return logits, loss
        return logits, None

# Define a class for the main Transformer model
# Define a class for the main Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
        self.decoder = Decoder(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)

    def forward(self, src_idx, tgt_idx, targets=None, target_mask=None):
        encoder_output = self.encoder(src_idx)
        logits, loss = self.decoder(tgt_idx, encoder_output, targets, target_mask)
        return logits, loss

    def evaluate(self, data_source, data_target, batch_size, pad_token):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(0, data_source.size(0), batch_size):
                batch_source = data_source[i:i+batch_size].to(self.device)
                batch_target = data_target[i:i+batch_size].to(self.device)
                source_mask = (batch_source != pad_token)
                target_mask = (batch_target != pad_token)
                _, loss = self(batch_source, batch_target, batch_target, target_mask)
                total_loss += loss.item()
        return total_loss / (data_source.size(0) // batch_size)
    
    def calculate_accuracy(self, data_source, data_target, batch_size, pad_token, eos_token):
        self.eval()
        total_accuracy = 0
        total_elements = 0
        with torch.no_grad():
            for i in range(0, data_source.size(0), batch_size):
                batch_source = data_source[i:i + batch_size].to(self.device)
                batch_target = data_target[i:i + batch_size].to(self.device)
                target_mask = (batch_target != pad_token)

                logits, _ = self(batch_source, batch_target, batch_target, target_mask)
                # Convert logits to predictions
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions = (predictions == batch_target) & target_mask
                total_accuracy += correct_predictions.sum().item()
                total_elements += target_mask.sum().item()

        return total_accuracy / total_elements if total_elements > 0 else 0

    def generate(self, src_idx, max_new_tokens, sos_token, eos_token):
        # src_idx is the source sequence (B, T_src)
        B, _ = src_idx.shape
        # target mask
        triangular_mask = torch.tril(torch.ones((max_new_tokens, max_new_tokens), device=self.device)).type(torch.bool)
        # Initialize target sequence with SOS token
        tgt_idx = torch.full((B, 1), sos_token, dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            # Get the predictions from the decoder
            logits, _ = self.forward(src_idx, tgt_idx, target_mask=triangular_mask)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running target sequence
            tgt_idx = torch.cat((tgt_idx, idx_next), dim=1)  # (B, T_tgt+1)

            # Check for EOS token
            if torch.any(idx_next == eos_token):
                break

        return tgt_idx
    

