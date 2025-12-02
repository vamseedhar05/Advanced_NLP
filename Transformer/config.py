import torch
config = {
    'name': 'final',
    'd_model': 512,
    'n_enc_layers': 3,  # Reduced for faster training
    'n_dec_layers': 3,  # Reduced for faster training
    'n_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_length': 360,  # Reduced for faster training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 0.0001,  # Reduced learning rate
    'clip': 1,
    'log_dir': 'logs',
    'weights_dir': 'weights',
    'save_interval': 3,
    'train_batch_size': 16,  # Increased batch size
    'val_batch_size': 16,
    'epochs': 10,  # Increased epochs
    'pos_encoding_type': 'relative'  # Start with absolute for simplicity
}