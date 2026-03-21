#!/usr/bin/env python3
"""Analyze and visualize token-classification training results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

PLOT_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']


def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_loss_curves(history, output_dir):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-o', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to {output_dir / 'loss_curves.png'}")


def plot_metrics(history, output_dir):
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(PLOT_METRICS):
        train_metric = [h['train_metrics'].get(metric, 0.0) for h in history]
        val_metric = [h['val_metrics'].get(metric, 0.0) for h in history]

        axes[idx].plot(epochs, train_metric, 'b-o', label='Train', linewidth=2)
        axes[idx].plot(epochs, val_metric, 'r-o', label='Validation', linewidth=2)
        axes[idx].set_xlabel('Epoch', fontsize=10)
        axes[idx].set_ylabel(metric.upper(), fontsize=10)
        axes[idx].set_title(f'{metric.upper()} over Epochs', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics curves to {output_dir / 'metrics_curves.png'}")


def plot_confusion_matrix(metrics, output_dir, split='val'):
    tp = metrics['tp']
    tn = metrics['tn']
    fp = metrics['fp']
    fn = metrics['fn']
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar_kws={'label': 'Count'},
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix ({split.capitalize()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{split}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_dir / f'confusion_matrix_{split}.png'}")


def generate_report(history, test_results, output_dir):
    report_file = output_dir / 'training_report.txt'
    best_epoch = max(history, key=lambda item: item['val_metrics'].get('f1', 0.0))

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('TOKEN CLASSIFICATION TRAINING REPORT\n')
        f.write('=' * 80 + '\n\n')

        f.write(f"Training completed with {len(history)} epochs\n\n")

        f.write('BEST VALIDATION PERFORMANCE:\n')
        f.write('-' * 80 + '\n')
        f.write(f"Epoch: {best_epoch['epoch']}\n")
        f.write(f"Validation Loss: {best_epoch['val_loss']:.4f}\n")
        f.write(f"Accuracy:  {best_epoch['val_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision: {best_epoch['val_metrics']['precision']:.4f}\n")
        f.write(f"Recall:    {best_epoch['val_metrics']['recall']:.4f}\n")
        f.write(f"F1 Score:  {best_epoch['val_metrics']['f1']:.4f}\n")
        f.write(f"AUC:       {best_epoch['val_metrics']['auc']:.4f}\n")
        f.write(f"MCC:       {best_epoch['val_metrics']['mcc']:.4f}\n")
        f.write(f"Specificity: {best_epoch['val_metrics']['specificity']:.4f}\n\n")

        f.write('Confusion Matrix:\n')
        f.write(f"  True Positives:  {best_epoch['val_metrics']['tp']}\n")
        f.write(f"  True Negatives:  {best_epoch['val_metrics']['tn']}\n")
        f.write(f"  False Positives: {best_epoch['val_metrics']['fp']}\n")
        f.write(f"  False Negatives: {best_epoch['val_metrics']['fn']}\n\n")

        if test_results:
            test_metrics = test_results.get('token_level', test_results)
            f.write('=' * 80 + '\n')
            f.write('TEST SET PERFORMANCE (TOKEN LEVEL):\n')
            f.write('-' * 80 + '\n')
            f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {test_metrics['f1']:.4f}\n")
            f.write(f"AUC:       {test_metrics['auc']:.4f}\n")
            f.write(f"MCC:       {test_metrics['mcc']:.4f}\n")
            f.write(f"Specificity: {test_metrics['specificity']:.4f}\n\n")

            f.write('Confusion Matrix:\n')
            f.write(f"  True Positives:  {test_metrics['tp']}\n")
            f.write(f"  True Negatives:  {test_metrics['tn']}\n")
            f.write(f"  False Positives: {test_metrics['fp']}\n")
            f.write(f"  False Negatives: {test_metrics['fn']}\n\n")

        f.write('=' * 80 + '\n')
        f.write('TRAINING PROGRESSION:\n')
        f.write('-' * 80 + '\n')
        f.write(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'Val F1':<12}{'Val AUC':<12}\n")
        f.write('-' * 80 + '\n')
        for h in history:
            f.write(
                f"{h['epoch']:<8}{h['train_loss']:<12.4f}{h['val_loss']:<12.4f}"
                f"{h['val_metrics']['f1']:<12.4f}{h['val_metrics']['auc']:<12.4f}\n"
            )
        f.write('\n' + '=' * 80 + '\n')

    print(f"Saved training report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze token classification training results')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory containing training results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    history_file = output_dir / 'training_history.json'
    if not history_file.exists():
        print(f"Error: {history_file} not found")
        return

    history = load_json(history_file)
    test_results_file = output_dir / 'test_results.json'
    test_results = load_json(test_results_file) if test_results_file.exists() else None

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print('Generating plots...')
    plot_loss_curves(history, plots_dir)
    plot_metrics(history, plots_dir)

    best_epoch = max(history, key=lambda item: item['val_metrics'].get('f1', 0.0))
    plot_confusion_matrix(best_epoch['val_metrics'], plots_dir, 'val')

    if test_results:
        plot_confusion_matrix(test_results.get('token_level', test_results), plots_dir, 'test')

    print('Generating report...')
    generate_report(history, test_results, output_dir)

    print('\nAnalysis complete!')
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main()
