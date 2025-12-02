import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
import argparse
import random
import numpy as np
from model import Transformer
from config import config
from dataset import EUbookshopFi2En
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import pandas as pd
from tqdm import tqdm

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def greedy_decode(model, src, src_mask, max_len, device, trg_vocab):
    """Greedy decoding: always select the token with the highest probability at each step."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Encode the source
        enc_output = model.encoder(src, src_mask)
        
        # Initialize target with <bos>
        trg_indexes = [EUbookshopFi2En.BOS_IDX]
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor).to(device)
            
            # Decode
            output = model.decoder(trg_tensor, enc_output, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            
            if pred_token == EUbookshopFi2En.EOS_IDX:
                break
                
        return trg_indexes

def beam_search_decode(model, src, src_mask, max_len, device, trg_vocab, beam_width=5):
    """Beam search decoding: maintain the top-B sequences (beams) at each step."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Encode the source
        enc_output = model.encoder(src, src_mask)
        
        # Initialize beams: (sequence, score)
        beams = [([EUbookshopFi2En.BOS_IDX], 0)]
        
        for i in range(max_len):
            all_candidates = []
            
            # Expand each beam
            for seq, score in beams:
                # If the beam ends with EOS, keep it as is
                if seq[-1] == EUbookshopFi2En.EOS_IDX:
                    all_candidates.append((seq, score))
                    continue
                
                # Prepare input for decoder
                trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(trg_tensor).to(device)
                
                # Get predictions
                output = model.decoder(trg_tensor, enc_output, trg_mask, src_mask)
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
                
                # Create new candidates
                for j in range(beam_width):
                    next_token = topk_indices[0, j].item()
                    new_score = score + topk_probs[0, j].item()
                    new_seq = seq + [next_token]
                    all_candidates.append((new_seq, new_score))
            
            # Select top beams
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all beams end with EOS
            if all(beam[0][-1] == EUbookshopFi2En.EOS_IDX for beam in beams):
                break
        
        # Return the best beam
        return beams[0][0]

def top_k_decode(model, src, src_mask, max_len, device, trg_vocab, k=10, temperature=1.0):
    """Top-k sampling: sample the next token from the top-k most probable tokens."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Encode the source
        enc_output = model.encoder(src, src_mask)
        
        # Initialize target with <bos>
        trg_indexes = [EUbookshopFi2En.BOS_IDX]
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor).to(device)
            
            # Get predictions
            output = model.decoder(trg_tensor, enc_output, trg_mask, src_mask)
            logits = output[:, -1, :] / temperature
            
            # Apply top-k filtering
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
            topk_probs = F.softmax(topk_logits, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(topk_probs, 1).item()
            next_token = topk_indices[0, next_token].item()
            
            trg_indexes.append(next_token)
            
            if next_token == EUbookshopFi2En.EOS_IDX:
                break
                
        return trg_indexes

def translate_sentence(sentence: Union[List[str], str], model: Transformer, 
                      src_tokenizer, trg_tokenizer, max_len=50, device='cpu', 
                      method='greedy', **kwargs):
    """Translate a sentence using the specified decoding method."""
    # Tokenize the source sentence
    if isinstance(sentence, str):
        tokens = src_tokenizer.encode(sentence).tokens
    else:
        tokens = sentence
    
    # Convert tokens to indices
    src_indices = [src_tokenizer.token_to_id(token) for token in tokens]
    src_indices = [EUbookshopFi2En.BOS_IDX] + src_indices + [EUbookshopFi2En.EOS_IDX]
    
    # Create source tensor and mask
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor).to(device)
    
    # Choose decoding method
    if method == 'greedy':
        trg_indices = greedy_decode(model, src_tensor, src_mask, max_len, device, trg_tokenizer)
    elif method == 'beam_search':
        trg_indices = beam_search_decode(model, src_tensor, src_mask, max_len, device, trg_tokenizer, 
                                        beam_width=kwargs.get('beam_width', 5))
    elif method == 'top_k':
        trg_indices = top_k_decode(model, src_tensor, src_mask, max_len, device, trg_tokenizer, 
                                  k=kwargs.get('k', 10), temperature=kwargs.get('temperature', 1.0))
    else:
        raise ValueError(f"Unknown decoding method: {method}")
    
    # Convert indices to tokens
    trg_tokens = [trg_tokenizer.id_to_token(idx) for idx in trg_indices]
    
    # Remove special tokens for final output
    trg_tokens = [token for token in trg_tokens if token not in EUbookshopFi2En.SPECIAL_SYMBOLS]
    
    return ' '.join(trg_tokens)

def evaluate_model(model, test_loader, device, method='greedy', **kwargs):
    """Evaluate model on test set and return BLEU score."""
    model.to(device)  # Ensure model is on the correct device
    model.eval()
    references = []
    hypotheses = []
    
    for batch in tqdm(test_loader, desc=f"Evaluating with {method}"):
        src_texts = batch['fi_text']
        trg_texts = batch['en_text']
        
        for src_text, trg_text in zip(src_texts, trg_texts):
            # Translate source text
            translation = translate_sentence(
                src_text, model, test_loader.dataset.fi_tokenizer, 
                test_loader.dataset.en_tokenizer, method=method, 
                device=device, **kwargs
            )
            
            # Prepare reference (remove special tokens)
            ref_tokens = [token for token in test_loader.dataset.en_tokenizer.encode(trg_text).tokens 
                         if token not in EUbookshopFi2En.SPECIAL_SYMBOLS]
            
            hypotheses.append(translation.split())
            references.append([ref_tokens])
    
    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    
    return bleu_score * 100  # Convert to percentage

def main():
    parser = argparse.ArgumentParser(description='Translate a sentence or evaluate models on test set')
    parser.add_argument('--sentence', type=str, default=None, help='Sentence to translate')
    parser.add_argument('--method', type=str, default='greedy', 
                       choices=['greedy', 'beam_search', 'top_k'], 
                       help='Decoding method to use')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--k', type=int, default=10, help='K for top-k sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--weights_path', type=str, default='weights/best_model.pt', 
                       help='Path to model weights')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on test set')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset to get tokenizers
    train_dataset = EUbookshopFi2En('train')
    test_dataset = EUbookshopFi2En('test')
    
    fi_tokenizer = train_dataset.fi_tokenizer
    en_tokenizer = train_dataset.en_tokenizer
    
    # Update config with vocabulary sizes
    config['src_vocab_size'] = fi_tokenizer.get_vocab_size()
    config['trg_vocab_size'] = en_tokenizer.get_vocab_size()
    config['src_pad_idx'] = EUbookshopFi2En.PAD_IDX
    config['trg_pad_idx'] = EUbookshopFi2En.PAD_IDX
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            collate_fn=EUbookshopFi2En.collate_fn)    
    if args.evaluate:
        # Define models to evaluate
        models = {
            'rope': 'weights/best_model_rope.pt',
            'relative': 'weights/best_model_relative.pt'
        }
        
        # Define decoding strategies to evaluate
        decoding_strategies = [
            {'method': 'greedy', 'name': 'Greedy'},
            {'method': 'beam_search', 'beam_width': 5, 'name': 'Beam Search (5)'},
            {'method': 'beam_search', 'beam_width': 10, 'name': 'Beam Search (10)'},
            {'method': 'top_k', 'k': 10, 'temperature': 1.0, 'name': 'Top-k (10, 1.0)'},
            {'method': 'top_k', 'k': 50, 'temperature': 1.0, 'name': 'Top-k (50, 1.0)'},
            {'method': 'top_k', 'k': 10, 'temperature': 0.8, 'name': 'Top-k (10, 0.8)'},
            {'method': 'top_k', 'k': 50, 'temperature': 0.8, 'name': 'Top-k (50, 0.8)'},
        ]
        
        results = []
        
        for model_name, weights_path in models.items():
            # Initialize model with appropriate positional encoding
            model = Transformer(
                config['src_vocab_size'],
                config['trg_vocab_size'],
                config['src_pad_idx'],
                config['trg_pad_idx'],
                config['d_model'],
                config['n_enc_layers'],
                config['n_dec_layers'],
                config['n_heads'],
                config['max_length'],
                config['d_ff'],
                config['dropout'],
                device=device,  # Use the device we determined
                pos_encoding_type='rope' if model_name == 'rope' else 'relative'
            )
            
            # Load weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            
            # Evaluate each decoding strategy
            for strategy in decoding_strategies:
                bleu_score = evaluate_model(
                    model, test_loader, device, 
                    **{k: v for k, v in strategy.items() if k != 'name'}
                )
                
                results.append({
                    'Model': model_name,
                    'Decoding Strategy': strategy['name'],
                    'BLEU Score': bleu_score
                })
                
                print(f"{model_name} with {strategy['name']}: BLEU = {bleu_score:.2f}")
        
        # Create results table
        df = pd.DataFrame(results)
        pivot_table = df.pivot(index='Decoding Strategy', columns='Model', values='BLEU Score')
        
        print("\n=== BLEU Scores Comparison ===")
        print(pivot_table.to_markdown())
        
        # Analysis
        print("\n=== Analysis ===")
        best_overall = df.loc[df['BLEU Score'].idxmax()]
        print(f"Best overall performance: {best_overall['Model']} with {best_overall['Decoding Strategy']} (BLEU: {best_overall['BLEU Score']:.2f})")
        
        # Compare positional encodings
        rope_avg = df[df['Model'] == 'rope']['BLEU Score'].mean()
        relative_avg = df[df['Model'] == 'relative']['BLEU Score'].mean()
        print(f"Average BLEU - RoPE: {rope_avg:.2f}, Relative: {relative_avg:.2f}")
        
        # Compare decoding strategies
        for strategy in decoding_strategies:
            strategy_name = strategy['name']
            strategy_scores = df[df['Decoding Strategy'] == strategy_name]['BLEU Score']
            print(f"{strategy_name} - Avg BLEU: {strategy_scores.mean():.2f}, Range: {strategy_scores.min():.2f}-{strategy_scores.max():.2f}")
    
    else:
        # Single sentence translation mode
        # Initialize model
        model = Transformer(
            config['src_vocab_size'],
            config['trg_vocab_size'],
            config['src_pad_idx'],
            config['trg_pad_idx'],
            config['d_model'],
            config['n_enc_layers'],
            config['n_dec_layers'],
            config['n_heads'],
            config['max_length'],
            config['d_ff'],
            config['dropout'],
            device=device,  # Use the device we determined
            pos_encoding_type='rope'  # Default for single translation
        )
        
        # Load weights
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
        model.to(device)
        
        # Translate the sentence
        translation = translate_sentence(
            args.sentence, 
            model, 
            fi_tokenizer, 
            en_tokenizer, 
            method=args.method,
            beam_width=args.beam_width,
            k=args.k,
            temperature=args.temperature,
            device=device
        )
        
        print(f'Input: {args.sentence}')
        print(f'Translation: {translation}')
        print(f'Method: {args.method}')
        
        if args.method == 'beam_search':
            print(f'Beam width: {args.beam_width}')
        elif args.method == 'top_k':
            print(f'K: {args.k}, Temperature: {args.temperature}')

if __name__ == '__main__':
    main()