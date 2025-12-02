import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
import time
import psutil
import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Shared utilities
from utils import (
    AGNewsDataset,
    get_model_memory_footprint,
    measure_inference_latency,
    evaluate,
    plot_confusion_matrix,
    plot_metrics_summary,
    save_metrics_report
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    model_name = "gpt2"  # GPT-2 small
    num_classes = 4  # AG News has 4 classes
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    warmup_steps = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./gpt2_ag_news_finetuned"
    num_inference_samples = 1000  # For latency measurement

config = Config()

# Custom GPT-2 Classification Model
class GPT2ForClassification(nn.Module):
    def __init__(self, num_classes):
        super(GPT2ForClassification, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt2.config.n_embd, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        last_hidden_states = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        pooled_output = self.dropout(last_hidden_states)
        logits = self.classifier(pooled_output)
        
        return logits

# Custom Dataset
# AGNewsDataset provided by utils.AGNewsDataset

# Load and prepare data
def load_data():
    print("Loading AG News dataset...")
    dataset = load_dataset('ag_news')
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    return train_texts, train_labels, test_texts, test_labels

# Memory measurement
# get_model_memory_footprint provided by utils
def get_gpu_memory():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def get_system_memory():
    """Get system RAM usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

# Inference latency measurement
# use measure_inference_latency from utils

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def save_metrics_report(metrics, save_path):
    # delegate to utils.save_metrics_report with default names
    from utils import save_metrics_report as _save
    _save(metrics, save_path)

# Main training loop
def main():
    print(f"Using device: {config.device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if config.save_path and os.path.exists(f"{config.save_path}/best_model.pt"):
        print(f"Loading model from {config.save_path}/best_model.pt")
        checkpoint = torch.load(f"{config.save_path}/best_model.pt", map_location=config.device)
        model = GPT2ForClassification(config.num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        print("Model loaded successfully.")
        return
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_data()
    
    # Create datasets
    train_dataset = AGNewsDataset(train_texts, train_labels, tokenizer, config.max_length)
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Initialize model
    print("\nInitializing GPT-2 model for classification...")
    model = GPT2ForClassification(config.num_classes).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Measure initial memory
    model_memory = get_model_memory_footprint(model)
    print(f"Model memory footprint: {model_memory:.2f} MB")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, config.device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Evaluate
        test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, config.device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            print(f"New best accuracy: {best_accuracy:.4f}. Saving model...")
            os.makedirs(config.save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, f"{config.save_path}/best_model.pt")
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION - COMPUTING ALL METRICS")
    print(f"{'='*80}")
    
    test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, config.device)
    
    # Compute performance metrics
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    # Measure memory during inference
    gpu_memory = get_gpu_memory()
    system_memory = get_system_memory()
    
    # Measure inference latency
    avg_latency, std_latency, latency_samples = measure_inference_latency(
        model, test_loader, config.device, config.num_inference_samples
    )
    
    # Compile all metrics
    metrics = {
        'model_name': 'GPT-2 Small (Full Fine-tuning)',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_memory_mb': model_memory,
        'gpu_memory_mb': gpu_memory,
        'system_memory_mb': system_memory,
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'latency_samples': latency_samples,
        'accuracy': test_acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'class_names': class_names,
        'classification_report': classification_report(true_labels, predictions, 
                                                       target_names=class_names)
    }
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(true_labels, predictions, class_names, config.save_path)
    plot_metrics_summary(metrics, config.save_path)
    
    # Save comprehensive report
    save_metrics_report(metrics, config.save_path)
    
    print(f"\n{'='*80}")
    print("All metrics computed and saved successfully!")
    print(f"Results saved to: {config.save_path}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()