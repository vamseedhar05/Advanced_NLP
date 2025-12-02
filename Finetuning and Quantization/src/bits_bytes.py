import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
import time
import psutil
import os
import json
import copy
from sklearn.metrics import precision_recall_fscore_support, classification_report

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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    model_name = "gpt2"
    num_classes = 4
    max_length = 128
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_model_path = "./gpt2_ag_news_finetuned/best_model.pt"
    save_path_8bit = "./gint8_bnb"
    save_path_4bit = "./gnf4_bnb"
    num_inference_samples = 1000

config = Config()

# ============================================================================
# GPT-2 CLASSIFICATION MODEL
# ============================================================================

class GPT2ForClassification(nn.Module):
    def __init__(self, num_classes, quantization_config=None):
        super(GPT2ForClassification, self).__init__()
        if quantization_config:
            self.gpt2 = GPT2Model.from_pretrained(
                config.model_name, 
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.gpt2 = GPT2Model.from_pretrained(config.model_name)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt2.config.n_embd, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        last_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        pooled_output = self.dropout(last_hidden_states)
        
        # CRITICAL FIX: Convert to float32 if needed (for quantized models that output float16)
        if pooled_output.dtype != self.classifier.weight.dtype:
            pooled_output = pooled_output.to(self.classifier.weight.dtype)
        
        logits = self.classifier(pooled_output)
        
        return logits


def save_metrics_report(metrics, save_path, baseline_memory=None):
    from utils import save_metrics_report as _save
    extra_info = {'baseline_memory_mb': baseline_memory} if baseline_memory is not None else None
    _save(metrics, save_path, extra_info=extra_info)

# ============================================================================
# QUANTIZATION FUNCTIONS
# ============================================================================

def load_and_quantize_model(baseline_model_path, quantization_bits=8):
    """
    Load baseline model and apply bitsandbytes quantization
    CRITICAL: We need to load the FINE-TUNED weights, then apply quantization
    
    Args:
        baseline_model_path: Path to the fine-tuned baseline model
        quantization_bits: 8 for INT8, 4 for NF4
    
    Returns:
        quantized_model: Model with quantized GPT-2 backbone
    """
    print(f"\nLoading fine-tuned baseline model from {baseline_model_path}...")
    
    # Load baseline checkpoint
    checkpoint = torch.load(baseline_model_path, map_location='cpu')
    baseline_accuracy = checkpoint['accuracy']
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # STEP 1: Load the FULL fine-tuned model first (without quantization)
    print("\nStep 1: Loading fine-tuned model (FP32)...")
    temp_model = GPT2ForClassification(config.num_classes, quantization_config=None)
    temp_model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Fine-tuned weights loaded successfully")
    
    # STEP 2: Save the fine-tuned GPT-2 weights temporarily
    print("\nStep 2: Extracting fine-tuned GPT-2 weights...")
    finetuned_gpt2_state = temp_model.gpt2.state_dict()
    classifier_weight = temp_model.classifier.weight.data.clone()
    classifier_bias = temp_model.classifier.bias.data.clone()
    del temp_model  # Free memory
    
    # Create quantization config
    if quantization_bits == 8:
        print("\nStep 3: Applying 8-bit quantization with bitsandbytes...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model_name_str = "INT8 (bitsandbytes)"
    elif quantization_bits == 4:
        print("\nStep 3: Applying 4-bit NF4 quantization with bitsandbytes...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_name_str = "NF4 (bitsandbytes)"
    else:
        raise ValueError("quantization_bits must be 4 or 8")
    
    # STEP 3: Create new model with quantization from the FINE-TUNED weights
    # BitsAndBytes can't quantize an existing model, so we need a workaround:
    # We'll save the fine-tuned GPT-2 locally and load it with quantization
    
    print("Saving fine-tuned GPT-2 temporarily for quantization...")
    temp_gpt2_path = "./temp_finetuned_gpt2"
    os.makedirs(temp_gpt2_path, exist_ok=True)
    
    # Save the fine-tuned GPT-2
    temp_save_model = GPT2ForClassification(config.num_classes, quantization_config=None)
    temp_save_model.load_state_dict(checkpoint['model_state_dict'])
    temp_save_model.gpt2.save_pretrained(temp_gpt2_path)
    del temp_save_model
    
    # Now load the fine-tuned GPT-2 WITH quantization
    print(f"Loading fine-tuned GPT-2 with {model_name_str} quantization...")
    from transformers import GPT2Model
    quantized_gpt2 = GPT2Model.from_pretrained(
        temp_gpt2_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # ensure the backbone parameters live on a single GPU (if available)
    if torch.cuda.is_available():
        try:
            quantized_gpt2 = quantized_gpt2.to("cuda:0")
        except Exception:
            pass
    
    # Create the classification model with the quantized GPT-2
    model = GPT2ForClassification.__new__(GPT2ForClassification)
    nn.Module.__init__(model)
    model.gpt2 = quantized_gpt2
    model.dropout = nn.Dropout(0.1)
    model.classifier = nn.Linear(model.gpt2.config.n_embd, config.num_classes)
    
    # Load classifier weights
    print("Loading fine-tuned classifier weights...")
    with torch.no_grad():
        model.classifier.weight.copy_(classifier_weight)
        model.classifier.bias.copy_(classifier_bias)
    
    # Move non-quantized layers to the same device as quantized model
    target_device = next(model.gpt2.parameters()).device
    print(f"Moving classifier and dropout to device: {target_device}")
    
    model.classifier = model.classifier.to(target_device)
    model.dropout = model.dropout.to(target_device)
    
    # For 4-bit models, convert classifier to float16
    if quantization_bits == 4:
        print("Converting classifier to float16 for 4-bit quantization compatibility")
        model.classifier = model.classifier.half()
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_gpt2_path, ignore_errors=True)
    
    print(f"\n{model_name_str} quantization applied successfully!")
    print(f"Model configuration:")
    print(f"  - GPT-2 backbone: {next(model.gpt2.parameters()).device}, dtype: {next(model.gpt2.parameters()).dtype}")
    print(f"  - Classifier: {next(model.classifier.parameters()).device}, dtype: {next(model.classifier.parameters()).dtype}")
    
    return model, baseline_accuracy

def evaluate_quantized_model(model, test_loader, device, save_path, model_name, baseline_memory=None):
    """
    Evaluate quantized model and save all metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*80}")
    
    # For quantized models with device_map="auto", we don't move the model
    # It's already on the correct device. We just need to ensure inputs go to the right device
    
    # Evaluate
    test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, 'cuda' if torch.cuda.is_available() else device)
    
    # Compute metrics
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    # Measure latency
    avg_latency, std_latency, latency_samples = measure_inference_latency(
        model, test_loader, 'cuda' if torch.cuda.is_available() else device, config.num_inference_samples
    )
    
    # Get model memory
    model_memory = get_model_memory_footprint(model)
    
    total_params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    
    # Compile metrics
    metrics = {
        'model_name': model_name,
        'total_params': total_params,
        'model_memory_mb': model_memory,
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
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    title_suffix = model_name
    plot_confusion_matrix(true_labels, predictions, class_names, save_path, title_suffix)
    plot_metrics_summary(metrics, save_path)
    
    # Save report
    save_metrics_report(metrics, save_path, baseline_memory)
    
    print(f"\n{'='*80}")
    print("All metrics computed and saved successfully!")
    print(f"Results saved to: {save_path}/")
    print(f"{'='*80}\n")
    
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Using device: {config.device}")
    
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. bitsandbytes quantization requires GPU.")
        print("The code will still run but may not show the expected memory benefits.")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    print("\nLoading AG News dataset...")
    dataset = load_dataset('ag_news')
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Create dataset and dataloader
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Get baseline memory for comparison
    print("\nLoading baseline model to get memory footprint...")
    baseline_model = GPT2ForClassification(config.num_classes)
    baseline_checkpoint = torch.load(config.baseline_model_path, map_location='cpu')
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_memory = get_model_memory_footprint(baseline_model)
    print(f"Baseline model memory: {baseline_memory:.2f} MB")
    del baseline_model  # Free memory
    
    # ========================================================================
    # 8-BIT QUANTIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: 8-BIT QUANTIZATION WITH BITSANDBYTES")
    print("="*80)
    
    model_8bit, baseline_acc = load_and_quantize_model(config.baseline_model_path, quantization_bits=8)
    
    metrics_8bit = evaluate_quantized_model(
        model_8bit, 
        test_loader, 
        config.device, 
        config.save_path_8bit,
        "GPT-2 Small (INT8 - bitsandbytes)",
        baseline_memory
    )
    
    # Free memory
    del model_8bit
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================================================
    # 4-BIT NF4 QUANTIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: 4-BIT NF4 QUANTIZATION WITH BITSANDBYTES")
    print("="*80)
    
    model_4bit, _ = load_and_quantize_model(config.baseline_model_path, quantization_bits=4)
    
    metrics_4bit = evaluate_quantized_model(
        model_4bit, 
        test_loader, 
        config.device, 
        config.save_path_4bit,
        "GPT-2 Small (NF4 - bitsandbytes)",
        baseline_memory
    )
    
    print("\n" + "="*80)
    print("ALL QUANTIZATION EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\n8-bit results: {config.save_path_8bit}/")
    print(f"4-bit results: {config.save_path_4bit}/")
    print("\n")

if __name__ == "__main__":
    main()