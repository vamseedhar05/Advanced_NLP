import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import random

from model import Transformer
from utils import AverageMeter
from config import config
from dataset import EUbookshopFi2En

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, config):
        # Configs & Parameters
        self.config = config
        self.src_vocab_size = self.config['src_vocab_size']
        self.trg_vocab_size = self.config['trg_vocab_size']
        self.d_ff = self.config['d_ff']
        self.d_model = self.config['d_model']
        self.n_enc_layers = self.config['n_enc_layers']
        self.n_dec_layers = self.config['n_dec_layers']
        self.n_heads = self.config['n_heads']
        self.max_length = self.config['max_length']
        self.dropout = self.config['dropout']
        self.device = self.config['device']
        self.src_pad_idx = self.config['src_pad_idx']
        self.trg_pad_idx = self.config['trg_pad_idx']
        self.lr = self.config['lr']
        self.clip = self.config['clip']
        self.log_dir = self.config['log_dir']
        self.pos_encoding_type = self.config.get('pos_encoding_type', 'absolute')

        self.model = Transformer(self.src_vocab_size,
                                 self.trg_vocab_size,
                                 self.src_pad_idx,
                                 self.trg_pad_idx,
                                 self.d_model,
                                 self.n_enc_layers,
                                 self.n_dec_layers,
                                 self.n_heads,
                                 self.max_length,
                                 self.d_ff,
                                 self.dropout,
                                 self.device,
                                 self.pos_encoding_type)
        self._init_weights()
        self.model.to(self.device)

        # Optimizer
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.98), 
            eps=1e-9,
            weight_decay=0.0001
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        # Loss Function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        self.criterion.to(self.device)

        # Metrics
        self.loss_tracker = AverageMeter('loss')

        # JSON logging
        self.loss_log = []
        os.makedirs(self.config['log_dir'], exist_ok=True)
        self.log_path = os.path.join(self.config['log_dir'], f"{self.config['name']}_loss_log.json")

    def _init_weights(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, dataloader, epoch, total_epochs):
        self.model.train()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Epoch: {epoch}/{total_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for batch in iterator:
                src, trg = batch['fi_tensor'].to(self.device), batch['en_tensor'].to(self.device)
                # src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.loss_tracker.update(loss.item())
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def evaluate(self, dataloader):
        self.model.eval()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.no_grad():
                for batch in iterator:
                    src, trg = batch['fi_tensor'].to(self.device), batch['en_tensor'].to(self.device)
                    output = self.model(src, trg[:, :-1])
                    output_dim = output.shape[-1]
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:, 1:].contiguous().view(-1)

                    loss = self.criterion(output, trg)
                    self.loss_tracker.update(loss.item())
                    avg_loss = self.loss_tracker.avg
                    iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def fit(self, train_loader, valid_loader, epochs):
        import json
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            print()
            train_loss = self.train(train_loader, epoch, epochs)
            val_loss = self.evaluate(valid_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log losses to JSON
            self.loss_log.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.config['weights_dir'], 'best_model.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved Best Model at {save_path} with val_loss: {val_loss:.4f}')
            
            # Save checkpoint periodically
            if epoch % self.config['save_interval'] == 0 or epoch == epochs:
                save_path = os.path.join(self.config['weights_dir'], f'epoch_{epoch}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved Model at {save_path}')
                
            with open(self.log_path, "w") as f:
                json.dump(self.loss_log, f, indent=2)


if __name__ == '__main__':
    batch_size = config['train_batch_size']

    # Create train and validation datasets
    train_dataset = EUbookshopFi2En('train', 
                                   fi_path='EUbookshop/EUbookshop.fi', 
                                   en_path='EUbookshop/EUbookshop.en')
    valid_dataset = EUbookshopFi2En('valid', 
                                   fi_path='EUbookshop/EUbookshop.fi', 
                                   en_path='EUbookshop/EUbookshop.en')

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=EUbookshopFi2En.collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,  # Typically don't shuffle validation
                              collate_fn=EUbookshopFi2En.collate_fn)

    # Get vocabulary sizes from the tokenizers
    config['src_vocab_size'] = train_dataset.fi_tokenizer.get_vocab_size()
    config['trg_vocab_size'] = train_dataset.en_tokenizer.get_vocab_size()
    config['src_pad_idx'] = EUbookshopFi2En.PAD_IDX
    config['trg_pad_idx'] = EUbookshopFi2En.PAD_IDX
    
    trainer = Trainer(config)
    trainer.fit(train_loader, valid_loader, config['epochs'])