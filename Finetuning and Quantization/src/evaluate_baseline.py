import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np

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


# Configuration
class Config:
    model_name = "gpt2"  # GPT-2 small
    num_classes = 4  # AG News has 4 classes
    max_length = 128
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetuned_model_path = "./gpt2_ag_news_finetuned/best_model.pt"
    save_path = "./baseline_results"
    num_inference_samples = 1000

config = Config()


# Minimal GPT2 classification wrapper (same shape as fine-tuned model)
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

        x = self.dropout(last_hidden_states)
        logits = self.classifier(x)
        return logits


def main():
    os.makedirs(config.save_path, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Check model exists
    if not os.path.exists(config.finetuned_model_path):
        print(f"Finetuned model not found at {config.finetuned_model_path}. Please provide a trained model.")
        return

    # Load dataset
    print("Loading AG News dataset test split...")
    from datasets import load_dataset
    dataset = load_dataset('ag_news')
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load model
    print(f"Loading finetuned model from {config.finetuned_model_path}...")
    checkpoint = torch.load(config.finetuned_model_path, map_location=config.device)
    model = GPT2ForClassification(config.num_classes)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(config.device)
    print("Model loaded.")

    # Evaluate
    print("Evaluating model on test set...")
    avg_loss, accuracy, predictions, true_labels = evaluate(model, test_loader, config.device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")

    # Compute per-class metrics
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )

    # Memory footprint
    model_memory_mb = get_model_memory_footprint(model)
    print(f"Model memory footprint: {model_memory_mb:.2f} MB")

    # Latency
    print("Measuring inference latency (this may take a while)...")
    avg_latency_ms, std_latency_ms, latency_samples = measure_inference_latency(
        model, test_loader, config.device, config.num_inference_samples
    )
    print(f"Avg latency: {avg_latency_ms:.2f} ms (std: {std_latency_ms:.2f} ms)")

    # Build metrics dict similar to compression scripts
    total_params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())

    metrics = {
        'model_name': 'GPT-2 Small (Finetuned Baseline)',
        'total_params': int(total_params),
        'model_memory_mb': float(model_memory_mb),
        'accuracy': float(accuracy),
        'loss': float(avg_loss),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'class_names': class_names,
        'classification_report': classification_report(true_labels, predictions, target_names=class_names),
        'avg_latency_ms': float(avg_latency_ms),
        'std_latency_ms': float(std_latency_ms),
        'latency_samples': latency_samples
    }

    # Visualizations and reporting
    print("Generating confusion matrix and metrics summary...")
    plot_confusion_matrix(true_labels, predictions, class_names, config.save_path)
    plot_metrics_summary(metrics, config.save_path)

    # Save metrics
    save_metrics_report(metrics, config.save_path)

    # Save a small artifact with model info
    artifact = {
        'model_path': config.finetuned_model_path,
        'model_name': metrics['model_name'],
        'accuracy': metrics['accuracy']
    }
    with open(os.path.join(config.save_path, 'artifact.json'), 'w') as f:
        import json
        json.dump(artifact, f, indent=2)

    print(f"Evaluation complete. Results saved to {config.save_path}")


if __name__ == '__main__':
    main()
