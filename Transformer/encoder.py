# encoder.py
import torch
import torch.nn as nn
from utils import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, mask):
        # Self-attention
        attn_output = self.self_attn(enc_input, enc_input, enc_input, mask)
        enc_input = self.norm1(enc_input + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(enc_input)
        enc_input = self.norm2(enc_input + self.dropout(ff_output))
        
        return enc_input

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, enc_layers, n_heads, max_length, d_ff, dropout, device, pos_encoding_type='absolute'):
        super().__init__()
        self.device = device
        self.n_layers = enc_layers
        self.scale = d_model ** 0.5
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        
        # Only use absolute positional embeddings if not using RoPE or relative
        if pos_encoding_type == 'absolute':
            self.pos_embedding = nn.Embedding(max_length, d_model)
        else:
            self.pos_embedding = None
            
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                self_attn=MultiHeadAttention(d_model, n_heads, dropout, pos_encoding_type, max_length),
                feed_forward=nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                ),
                dropout=dropout
            ) for _ in range(enc_layers)
        ])

    def forward(self, src, mask):
        N, seq_len = src.shape
        
        # Token embeddings
        tok_embeddings = self.tok_embedding(src) * self.scale
        
        # Position embeddings (only for absolute)
        if self.pos_embedding is not None:
            positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
            pos_embeddings = self.pos_embedding(positions)
            out = self.dropout(tok_embeddings + pos_embeddings)
        else:
            out = self.dropout(tok_embeddings)

        for block in self.blocks:
            out = block(out, mask)

        return out