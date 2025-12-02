# decoder.py
import torch
import torch.nn as nn
from utils import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, dec_input, enc_output, trg_mask, src_mask):
        # Self-attention
        self_attn_output = self.self_attn(dec_input, dec_input, dec_input, trg_mask)
        dec_input = self.norm1(dec_input + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output = self.cross_attn(dec_input, enc_output, enc_output, src_mask)
        dec_input = self.norm2(dec_input + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(dec_input)
        dec_input = self.norm3(dec_input + self.dropout(ff_output))
        
        return dec_input

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, dec_layers, n_heads, max_length, d_ff, dropout, device, pos_encoding_type='absolute'):
        super().__init__()
        self.device = device
        self.n_layers = dec_layers
        self.scale = d_model ** 0.5
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        
        # Only use absolute positional embeddings if not using RoPE or relative
        if pos_encoding_type == 'absolute':
            self.pos_embedding = nn.Embedding(max_length, d_model)
        else:
            self.pos_embedding = None
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                self_attn=MultiHeadAttention(d_model, n_heads, dropout, pos_encoding_type, max_length),
                cross_attn=MultiHeadAttention(d_model, n_heads, dropout, 'absolute', max_length),  # Cross attention always uses absolute
                feed_forward=nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                ),
                dropout=dropout
            ) for _ in range(dec_layers)
        ])

    def forward(self, dec_input, enc_output, trg_mask, src_mask):
        N, trg_len = dec_input.shape
        
        # Token embeddings
        tok_embeddings = self.tok_embedding(dec_input) * self.scale
        
        # Position embeddings (only for absolute)
        if self.pos_embedding is not None:
            positions = torch.arange(0, trg_len).expand(N, trg_len).to(self.device)
            pos_embeddings = self.pos_embedding(positions)
            dec_input = self.dropout(tok_embeddings + pos_embeddings)
        else:
            dec_input = self.dropout(tok_embeddings)

        for block in self.blocks:
            dec_input = block(dec_input, enc_output, trg_mask, src_mask)

        output = self.fc(dec_input)
        return output