class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length=5000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        # x: [batch_size, n_heads, seq_len, head_dim]
        batch_size, n_heads, seq_len, head_dim = x.shape
        position = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Calculate sinusoids
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        
        # Reshape to match the expected dimensions
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        
        # Expand to match batch size and number of heads
        sin = sin.expand(batch_size, n_heads, -1, -1)
        cos = cos.expand(batch_size, n_heads, -1, -1)
        
        # Apply rotary embeddings
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated_x

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_length=512):
        super().__init__()
        self.num_heads = num_heads
        self.max_length = max_length
        self.relative_attention_bias = nn.Embedding(2 * max_length + 1, num_heads)
        
    def forward(self, seq_len_q, seq_len_k):
        # Get the device from the embedding parameters
        device = self.relative_attention_bias.weight.device
        
        # Generate relative positions on the correct device
        context_position = torch.arange(seq_len_q, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len_k, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.clamp(relative_position, -self.max_length, self.max_length)
        relative_position += self.max_length
        
        # Get bias values
        bias = self.relative_attention_bias(relative_position)
        bias = bias.permute(2, 0, 1).contiguous()  # [num_heads, seq_len_q, seq_len_k]
        return bias.unsqueeze(0)  # [1, num_heads, seq_len_q, seq_len_k]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, pos_encoding_type='absolute', max_length=512):
        super().__init__()
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = self.head_dim ** 0.5
        self.pos_encoding_type = pos_encoding_type

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize positional encoding
        if pos_encoding_type == 'rope':
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_length)
        elif pos_encoding_type == 'relative':
            self.relative_bias = RelativePositionBias(n_heads, max_length)

    def forward(self, q, k, v, mask=None):
        N = q.size(0)          # batch_size
        Q = self.queries(q)    # shape: [N, query_len, embed_dim]
        K = self.keys(k)       # shape: [N, key_len, embed_dim]
        V = self.values(v)     # shape: [N, value_len, embed_dim]

        Q = Q.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, query_len, head_dim]
        K = K.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, key_len, head_dim]
        V = V.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, value_len, head_dim]

        # Apply rotary positional embeddings if enabled
        if self.pos_encoding_type == 'rope':
            Q = self.rope(Q)
            K = self.rope(K)

        # Calculate attention scores
        energy = (Q @ K.permute(0, 1, 3, 2)) / self.scale
        
        # Add relative position bias if enabled
        if self.pos_encoding_type == 'relative':
            seq_len_q, seq_len_k = Q.size(2), K.size(2)
            relative_bias = self.relative_bias(seq_len_q, seq_len_k)
            # Ensure relative_bias is on the same device as energy
            relative_bias = relative_bias.to(energy.device)
            energy = energy + relative_bias
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)

        attention = energy.softmax(-1)           # shape: [N, n_heads, query_len, key_len]
        x = self.dropout(attention) @ V          # shape: [N, n_heads, query_len, key_len]
        x = x.permute(0, 2, 1, 3).contiguous()   # shape: [N, query_len, n_heads, head_dim]
        x = x.view(N, -1, self.embed_dim)        # shape: [N, query_len, embed_dim]
        x = self.proj(x)

        return x