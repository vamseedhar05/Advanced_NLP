import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Literal, Dict, Any
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import os

class EUbookshopFi2En(Dataset):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, split: Literal['train', 'valid', 'test'] = 'train',
                 fi_path: str = 'EUbookshop/EUbookshop.fi',
                 en_path: str = 'EUbookshop/EUbookshop.en',
                 max_samples: int = None,
                 vocab_size: int = 30000):
        super().__init__()
        self.split = split
        self.fi_path = fi_path
        self.en_path = en_path
        self.fi_texts, self.en_texts = self._load_data(max_samples)
        
        # Create or load tokenizers
        self.fi_tokenizer = self._build_tokenizer(self.fi_texts, 'fi', vocab_size)
        self.en_tokenizer = self._build_tokenizer(self.en_texts, 'en', vocab_size)

    def _load_data(self, max_samples=None):
        with open(self.fi_path, encoding='utf-8') as f_fi, open(self.en_path, encoding='utf-8') as f_en:
            fi_lines = [line.strip() for line in f_fi]
            en_lines = [line.strip() for line in f_en]
        assert len(fi_lines) == len(en_lines), "Mismatch in number of lines between Finnish and English files."
        
        # For test set, we might want to limit samples for quick evaluation
        if max_samples:
            fi_lines = fi_lines[:max_samples]
            en_lines = en_lines[:max_samples]
            
        total_lines = len(fi_lines)

        if self.split == 'train':
            split_idx = int(0.7 * total_lines)
            fi_lines, en_lines = fi_lines[:split_idx], en_lines[:split_idx]

        elif self.split == 'valid':
            start_idx = int(0.7 * total_lines)
            end_idx = int(0.8 * total_lines)
            fi_lines, en_lines = fi_lines[start_idx:end_idx], en_lines[start_idx:end_idx]

        elif self.split == 'test':
            split_idx = int(0.8 * total_lines)
            fi_lines, en_lines = fi_lines[split_idx:], en_lines[split_idx:]
            
        return fi_lines, en_lines

    def _build_tokenizer(self, texts, lang, vocab_size):
        tokenizer_path = f"tokenizer_{lang}.json"
        
        # Load tokenizer if it exists
        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Create a new tokenizer
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Trainer
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=self.SPECIAL_SYMBOLS,
                min_frequency=2
            )
            
            # Train the tokenizer
            tokenizer.train_from_iterator(texts, trainer=trainer)
            
            # Set post-processing
            tokenizer.post_processor = processors.TemplateProcessing(
                single="<bos> $A <eos>",
                special_tokens=[
                    ("<bos>", self.BOS_IDX),
                    ("<eos>", self.EOS_IDX),
                ],
            )
            
            # Save the tokenizer
            tokenizer.save(tokenizer_path)
        
        return tokenizer

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, index):
        fi_text = self.fi_texts[index]
        en_text = self.en_texts[index]
        
        # Encode texts
        fi_encoding = self.fi_tokenizer.encode(fi_text)
        en_encoding = self.en_tokenizer.encode(en_text)
        
        # Convert to tensors
        fi_tensor = torch.tensor(fi_encoding.ids, dtype=torch.long)
        en_tensor = torch.tensor(en_encoding.ids, dtype=torch.long)
        
        # Return both tensors and raw text for evaluation
        return {
            'fi_tensor': fi_tensor,
            'en_tensor': en_tensor,
            'fi_text': fi_text,
            'en_text': en_text
        }

    @classmethod
    def collate_fn(cls, batch):
        fi_batch, en_batch = [], []
        fi_texts, en_texts = [], []
        
        for item in batch:
            fi_batch.append(item['fi_tensor'])
            en_batch.append(item['en_tensor'])
            fi_texts.append(item['fi_text'])
            en_texts.append(item['en_text'])
        
        # Pad sequences
        fi_batch = pad_sequence(fi_batch, padding_value=cls.PAD_IDX, batch_first=True)
        en_batch = pad_sequence(en_batch, padding_value=cls.PAD_IDX, batch_first=True)
        
        return {
            'fi_tensor': fi_batch,
            'en_tensor': en_batch,
            'fi_text': fi_texts,
            'en_text': en_texts
        }

    def get_vocab_info(self) -> Dict[str, Any]:
        """Return vocabulary information for both languages"""
        return {
            'fi_vocab_size': self.fi_tokenizer.get_vocab_size(),
            'en_vocab_size': self.en_tokenizer.get_vocab_size(),
            'special_symbols': self.SPECIAL_SYMBOLS,
            'pad_idx': self.PAD_IDX,
            'bos_idx': self.BOS_IDX,
            'eos_idx': self.EOS_IDX,
            'unk_idx': self.UNK_IDX
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # Create datasets for all splits
    train_dataset = EUbookshopFi2En('train', vocab_size=30000)
    valid_dataset = EUbookshopFi2En('valid', vocab_size=30000)
    test_dataset = EUbookshopFi2En('test', vocab_size=30000)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=EUbookshopFi2En.collate_fn)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print('Finnish batch shape:', batch['fi_tensor'].shape)
    print('English batch shape:', batch['en_tensor'].shape)
    print('Finnish sample text:', batch['fi_text'][0])
    print('English sample text:', batch['en_text'][0])
    
    # Get vocabulary info
    vocab_info = test_dataset.get_vocab_info()
    print("Vocabulary info:", vocab_info)
    
    # Check sequence lengths
    fi_max_len = max(len(test_dataset.fi_tokenizer.encode(text).ids) for text in test_dataset.fi_texts)
    en_max_len = max(len(test_dataset.en_tokenizer.encode(text).ids) for text in test_dataset.en_texts)
    print("Max Finnish sequence length:", fi_max_len)
    print("Max English sequence length:", en_max_len)
    
    print('done')