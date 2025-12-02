import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

# model.py - Update the Transformer class
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, d_model, n_enc_layers, n_dec_layers, 
                 n_heads, max_length, d_ff, dropout, device, pos_encoding_type='absolute'):
        super().__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
        self.encoder = Encoder(src_vocab_size, d_model, n_enc_layers, n_heads, max_length, d_ff, dropout, device, pos_encoding_type)
        self.decoder = Decoder(trg_vocab_size, d_model, n_dec_layers, n_heads, max_length, d_ff, dropout, device, pos_encoding_type)
        
    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device) & trg_pad_mask.to(self.device)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, trg_mask, src_mask)
        
        return output


import torch
# from model import Transformer

if __name__ == "__main__":
    torch.random.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_size = 20
    trg_vocab_size = 20
    src_pad_idx = 0
    trg_pad_idx = 0
    d_model = 512
    n_enc_layers = 6
    n_dec_layers = 6
    n_heads = 8
    max_length = 100
    d_ff = 4
    dropout = 0.1

    # Dummy input data
    src = torch.randint(1, src_vocab_size, (2, 10)).to(device)
    trg = torch.randint(1, trg_vocab_size, (2, 10)).to(device)

    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
        d_model, n_enc_layers, n_dec_layers, n_heads, max_length, d_ff, dropout, device
    ).to(device)

    out = model(src, trg)
    print("Output shape:", out.shape)
    print("Output:", out)
