import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset
import numpy as np
import time
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
    save_path = "./gint8_scratch"
    num_inference_samples = 1000

config = Config()

# ============================================================================
# QUANTIZATION FUNCTIONS (From Scratch)
# ============================================================================

def quantize_tensor(tensor, num_bits=8):
    """
    Quantize a FP32 tensor to INT8 using symmetric quantization.
    
    Args:
        tensor: Input FP32 tensor
        num_bits: Number of bits for quantization (default: 8)
    
    Returns:
        quantized_tensor: INT8 tensor
        scale: Scale factor for dequantization
        zero_point: Zero point (0 for symmetric quantization)
    """
    # Get min and max values
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    
    # Calculate scale (symmetric quantization)
    max_val = torch.max(torch.abs(tensor))
    scale = max_val / qmax if max_val != 0 else 1.0
    
    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    
    return quantized.to(torch.int8), scale, 0

def dequantize_tensor(quantized_tensor, scale, zero_point=0):
    """
    Dequantize an INT8 tensor back to FP32.
    
    Args:
        quantized_tensor: INT8 tensor
        scale: Scale factor from quantization
        zero_point: Zero point from quantization
    
    Returns:
        dequantized_tensor: FP32 tensor
    """
    return (quantized_tensor.float() - zero_point) * scale

# ============================================================================
# QUANTIZED LINEAR AND CONV1D LAYERS
# ============================================================================

class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer that stores weights in INT8 and performs 
    computation in FP32 after dequantization.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weight storage
        self.register_buffer('quantized_weight', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        # Bias (kept in FP32)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def set_quantized_weight(self, weight):
        """Quantize and set the weight"""
        quantized, scale, _ = quantize_tensor(weight)
        self.quantized_weight = quantized
        # keep as tensor buffer so .to(device) works reliably
        self.weight_scale = torch.tensor(float(scale))
    
    def forward(self, x):
        # Dequantize weight on-the-fly
        weight_fp32 = dequantize_tensor(self.quantized_weight, self.weight_scale)
        return nn.functional.linear(x, weight_fp32, self.bias)

class QuantizedConv1D(nn.Module):
    """
    Quantized Conv1D layer (used in GPT-2) that stores weights in INT8.
    Conv1D in GPT-2 is actually a linear layer with transposed weights.
    """
    def __init__(self, nf, nx):
        super(QuantizedConv1D, self).__init__()
        self.nf = nf  # output features
        self.nx = nx  # input features
        
        # Quantized weight storage (note: Conv1D stores weights transposed)
        self.register_buffer('quantized_weight', torch.zeros((nx, nf), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        # Bias (kept in FP32)
        self.bias = nn.Parameter(torch.zeros(nf))
    
    def set_quantized_weight(self, weight):
        """Quantize and set the weight"""
        quantized, scale, _ = quantize_tensor(weight)
        self.quantized_weight = quantized
        self.weight_scale = scale
    
    def forward(self, x):
        # Dequantize weight on-the-fly
        weight_fp32 = dequantize_tensor(self.quantized_weight, self.weight_scale)
        
        # Conv1D forward pass (same as original GPT-2)
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight_fp32)
        x = x.view(size_out)
        return x

# ============================================================================
# GPT-2 CLASSIFICATION MODEL
# ============================================================================

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

# ============================================================================
# QUANTIZATION UTILITIES
# ============================================================================

def quantize_model_weights(model):
    """
    Quantize all Linear and Conv1D layers in the model to INT8.
    
    Args:
        model: The model to quantize
    
    Returns:
        quantized_model: Model with quantized weights
        quantization_stats: Statistics about quantization
    """
    from transformers.pytorch_utils import Conv1D  # Import GPT-2's Conv1D
    
    print("\nQuantizing model weights to INT8...")
    print("This will quantize ALL linear layers (Linear and Conv1D) in the GPT-2 backbone and classifier...")
    
    quantization_stats = {
        'total_layers': 0,
        'quantized_layers': 0,
        'original_size_mb': 0,
        'quantized_size_mb': 0,
        'compression_ratio': 0,
        'layer_types': {'Linear': 0, 'Conv1D': 0}
    }
    
    def quantize_linear_layers(module, prefix=''):
        from transformers.pytorch_utils import Conv1D
        
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Handle both nn.Linear and Conv1D (used in GPT-2)
            if isinstance(child, nn.Linear):
                quantization_stats['total_layers'] += 1
                quantization_stats['layer_types']['Linear'] += 1
                
                # Calculate original size
                original_size = child.weight.nelement() * child.weight.element_size()
                if child.bias is not None:
                    original_size += child.bias.nelement() * child.bias.element_size()
                quantization_stats['original_size_mb'] += original_size / 1024**2
                
                # Create quantized layer
                quantized_layer = QuantizedLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None)
                )
                
                # Quantize and set weights
                quantized_layer.set_quantized_weight(child.weight.data)
                
                # Copy bias if exists
                if child.bias is not None:
                    quantized_layer.bias.data = child.bias.data.clone()
                
                # Replace the layer
                setattr(module, name, quantized_layer)
                
                # Calculate quantized size (INT8 weights + FP32 bias + scale)
                quantized_size = quantized_layer.quantized_weight.nelement() * quantized_layer.quantized_weight.element_size()
                quantized_size += quantized_layer.weight_scale.nelement() * quantized_layer.weight_scale.element_size()
                if quantized_layer.bias is not None:
                    quantized_size += quantized_layer.bias.nelement() * quantized_layer.bias.element_size()
                quantization_stats['quantized_size_mb'] += quantized_size / 1024**2
                
                quantization_stats['quantized_layers'] += 1
                
                # Print progress
                if quantization_stats['quantized_layers'] % 10 == 0 or quantization_stats['quantized_layers'] <= 5:
                    print(f"  Quantized {quantization_stats['quantized_layers']} layers... Latest: {full_name} [Linear]")
                    
            elif isinstance(child, Conv1D):
                quantization_stats['total_layers'] += 1
                quantization_stats['layer_types']['Conv1D'] += 1
                
                # Calculate original size
                original_size = child.weight.nelement() * child.weight.element_size()
                original_size += child.bias.nelement() * child.bias.element_size()
                quantization_stats['original_size_mb'] += original_size / 1024**2
                
                # Create quantized Conv1D layer
                quantized_layer = QuantizedConv1D(child.nf, child.weight.shape[0])
                
                # Quantize and set weights
                quantized_layer.set_quantized_weight(child.weight.data)
                
                # Copy bias
                quantized_layer.bias.data = child.bias.data.clone()
                
                # Replace the layer
                setattr(module, name, quantized_layer)
                
                # Calculate quantized size
                quantized_size = quantized_layer.quantized_weight.nelement() * quantized_layer.quantized_weight.element_size()
                quantized_size += quantized_layer.weight_scale.nelement() * quantized_layer.weight_scale.element_size()
                quantized_size += quantized_layer.bias.nelement() * quantized_layer.bias.element_size()
                quantization_stats['quantized_size_mb'] += quantized_size / 1024**2
                
                quantization_stats['quantized_layers'] += 1
                
                # Print progress
                if quantization_stats['quantized_layers'] % 10 == 0 or quantization_stats['quantized_layers'] <= 5:
                    print(f"  Quantized {quantization_stats['quantized_layers']} layers... Latest: {full_name} [Conv1D]")
            else:
                # Recursively quantize child modules
                quantize_linear_layers(child, full_name)
    
    # Start quantization from the model root
    quantize_linear_layers(model)
    
    quantization_stats['compression_ratio'] = (
        quantization_stats['original_size_mb'] / quantization_stats['quantized_size_mb']
        if quantization_stats['quantized_size_mb'] > 0 else 0
    )
    
    print(f"\nQuantization complete!")
    print(f"  Total layers quantized: {quantization_stats['quantized_layers']}/{quantization_stats['total_layers']}")
    print(f"    - Linear layers: {quantization_stats['layer_types']['Linear']}")
    print(f"    - Conv1D layers: {quantization_stats['layer_types']['Conv1D']}")
    print(f"  Original size: {quantization_stats['original_size_mb']:.2f} MB")
    print(f"  Quantized size: {quantization_stats['quantized_size_mb']:.2f} MB")
    print(f"  Compression ratio: {quantization_stats['compression_ratio']:.2f}x")
    
    return model, quantization_stats

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Using device: {config.device}")
    
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
    
    # Load baseline model
    print(f"\nLoading baseline model from {config.baseline_model_path}...")
    model = GPT2ForClassification(config.num_classes)
    checkpoint = torch.load(config.baseline_model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    
    print(f"Baseline model loaded successfully!")
    print(f"  Baseline accuracy: {checkpoint['accuracy']:.4f}")
    
    # Quantize model
    baseline_memory = get_model_memory_footprint(model)
    model, quant_stats = quantize_model_weights(model)
    
    # Move to device after quantization
    model = model.to(config.device)
    
    # Evaluate quantized model
    print(f"\n{'='*80}")
    print("EVALUATING INT8 QUANTIZED MODEL")
    print(f"{'='*80}")
    
    test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, config.device)
    
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
        model, test_loader, config.device, config.num_inference_samples
    )
    
    # Get model memory
    model_memory = get_model_memory_footprint(model)
    memory_reduction = ((baseline_memory - model_memory) / baseline_memory) * 100
    
    total_params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    
    # Compile metrics
    metrics = {
        'model_name': 'GPT-2 Small (INT8 Quantized - From Scratch)',
        'total_params': total_params,
        'model_memory_mb': model_memory,
        'baseline_memory_mb': baseline_memory,
        'memory_reduction_percent': memory_reduction,
        'compression_ratio': quant_stats['compression_ratio'],
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
    os.makedirs(config.save_path, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(true_labels, predictions, class_names, config.save_path)
    plot_metrics_summary(metrics, config.save_path)
    
    # Save report
    save_metrics_report(metrics, config.save_path, quant_stats)
    
    # Save quantized model
    torch.save({
        'model_state_dict': model.state_dict(),
        'quantization_stats': quant_stats,
        'metrics': {k: v for k, v in metrics.items() 
                   if k not in ['latency_samples', 'classification_report']}
    }, f"{config.save_path}/quantized_model_int8_scratch.pt")
    
    print(f"\n{'='*80}")
    print("All metrics computed and saved successfully!")
    print(f"Results saved to: {config.save_path}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()