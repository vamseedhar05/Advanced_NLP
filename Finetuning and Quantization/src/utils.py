import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
import json


class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_model_memory_footprint(model):
    """Calculate model memory footprint in MB"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def measure_inference_latency(model, dataloader, device, num_samples=1000):
    """Measure average inference latency (ms) and return samples list"""
    model.eval()
    latencies = []
    samples_processed = 0

    print(f"\nMeasuring inference latency on {num_samples} samples...")

    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Warm-up
            if samples_processed == 0:
                _ = model(input_ids, attention_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Measure latency per sample
            for i in range(input_ids.size(0)):
                if samples_processed >= num_samples:
                    break

                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1]

                start_time = time.time()
                _ = model(single_input, single_mask)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
                samples_processed += 1

    avg_latency = np.mean(latencies) if len(latencies) > 0 else 0.0
    std_latency = np.std(latencies) if len(latencies) > 0 else 0.0

    return avg_latency, std_latency, latencies


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset. Returns (avg_loss, accuracy, predictions, true_labels)"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = accuracy_score(true_labels, predictions) if len(predictions) > 0 else 0.0

    return avg_loss, accuracy, predictions, true_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, filename='confusion_matrix.png', title=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    if title is None:
        title = 'Confusion Matrix'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {out_path}")
    plt.close()


def plot_metrics_summary(metrics, save_path, filename='metrics_summary.png'):
    """Generic metrics plotting (per-class bars, overall metrics, memory, latency)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Per-class metrics
    class_names = metrics.get('class_names', [])
    precision = metrics.get('precision_per_class', [])
    recall = metrics.get('recall_per_class', [])
    f1 = metrics.get('f1_per_class', [])

    x = np.arange(len(class_names))
    width = 0.25

    if len(class_names) > 0:
        axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Metrics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No per-class metrics', ha='center')

    # Overall metrics
    overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    overall_values = [
        metrics.get('accuracy', 0.0),
        metrics.get('precision_macro', 0.0),
        metrics.get('recall_macro', 0.0),
        metrics.get('f1_macro', 0.0)
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    axes[0, 1].bar(overall_metrics, overall_values, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Overall Metrics (Macro Average)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    for i, v in enumerate(overall_values):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Memory footprint
    memory_types = ['Model\nMemory']
    memory_values = [metrics.get('model_memory_mb', 0.0)]

    axes[1, 0].bar(memory_types, memory_values, color=['#9b59b6'], alpha=0.8)
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_title('Memory Footprint')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0, memory_values[0] + memory_values[0]*0.02, f'{memory_values[0]:.1f} MB', ha='center', fontweight='bold')

    # Inference latency distribution
    if 'latency_samples' in metrics and metrics['latency_samples']:
        axes[1, 1].hist(metrics['latency_samples'], bins=50, color='#34495e', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(metrics.get('avg_latency_ms', 0.0), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {metrics.get("avg_latency_ms", 0.0):.2f} ms')
        axes[1, 1].set_xlabel('Latency (ms)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Inference Latency Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No latency samples', ha='center')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Metrics summary saved to {out_path}")
    plt.close()


def save_metrics_report(metrics, save_path, quant_stats=None, extra_info=None, prefix='metrics_report'):
    """Save a simple text + json metrics report. extra_info can include baseline_memory etc."""
    os.makedirs(save_path, exist_ok=True)

    # Build a human-readable text report
    report_lines = []
    report_lines.append('='*80)
    report_lines.append(f"METRICS REPORT: {metrics.get('model_name','Model')}")
    report_lines.append('='*80)
    report_lines.append('\nOVERALL METRICS:')
    report_lines.append(f"  Accuracy: {metrics.get('accuracy',0.0):.4f}")
    report_lines.append(f"  Precision (macro): {metrics.get('precision_macro',0.0):.4f}")
    report_lines.append(f"  Recall (macro): {metrics.get('recall_macro',0.0):.4f}")
    report_lines.append(f"  F1 (macro): {metrics.get('f1_macro',0.0):.4f}")

    if 'class_names' in metrics:
        report_lines.append('\nPER-CLASS METRICS:')
        for i, cname in enumerate(metrics['class_names']):
            p = metrics.get('precision_per_class', [])[i] if metrics.get('precision_per_class') else 0.0
            r = metrics.get('recall_per_class', [])[i] if metrics.get('recall_per_class') else 0.0
            f = metrics.get('f1_per_class', [])[i] if metrics.get('f1_per_class') else 0.0
            report_lines.append(f"  {cname}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

    if extra_info is not None:
        report_lines.append('\nEXTRA INFO:')
        for k, v in extra_info.items():
            report_lines.append(f"  {k}: {v}")

    if quant_stats is not None:
        report_lines.append('\nQUANTIZATION STATISTICS:')
        for k, v in quant_stats.items():
            report_lines.append(f"  {k}: {v}")

    report_lines.append('\nCLASSIFICATION REPORT:')
    report_lines.append(metrics.get('classification_report', ''))

    text_report = '\n'.join(report_lines)
    text_path = os.path.join(save_path, f"{prefix}.txt")
    with open(text_path, 'w') as f:
        f.write(text_report)

    print(text_report)
    print(f"\nMetrics report saved to {text_path}")

    # Save JSON
    json_metrics = {k: v for k, v in metrics.items() if k not in ['latency_samples', 'classification_report']}
    if quant_stats is not None:
        json_metrics['quantization_stats'] = quant_stats
    if extra_info is not None:
        json_metrics['extra_info'] = extra_info

    # latency percentiles
    if 'latency_samples' in metrics and metrics['latency_samples']:
        json_metrics['latency_percentiles'] = {
            '50': float(np.percentile(metrics['latency_samples'], 50)),
            '95': float(np.percentile(metrics['latency_samples'], 95)),
            '99': float(np.percentile(metrics['latency_samples'], 99))
        }

    json_path = os.path.join(save_path, f"{prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)

    print(f"Metrics JSON saved to {json_path}")
