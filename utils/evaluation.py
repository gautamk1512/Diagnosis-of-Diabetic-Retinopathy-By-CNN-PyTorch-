import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
from datetime import datetime

class ModelEvaluator:
    """
    Comprehensive model evaluation for Diabetic Retinopathy Classification
    """
    def __init__(self, model, device, class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names or ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        self.num_classes = len(self.class_names)
    
    def evaluate_model(self, data_loader, save_results=True, results_dir='results'):
        """
        Comprehensive model evaluation
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        if save_results:
            self.save_evaluation_results(metrics, all_targets, all_predictions, 
                                       all_probabilities, results_dir)
        
        return metrics, all_targets, all_predictions, all_probabilities
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (for multiclass)
        try:
            if self.num_classes > 2:
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_prob, save_path=None):
        """
        Plot ROC curves for multiclass classification
        """
        if self.num_classes <= 2:
            return
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multiclass Classification')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_true, y_prob, save_path=None):
        """
        Plot Precision-Recall curves for multiclass classification
        """
        if self.num_classes <= 2:
            return
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Plot PR curve for each class
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{self.class_names[i]} (AUC = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Multiclass Classification')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """
        Plot class distribution comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        true_counts = np.bincount(y_true, minlength=self.num_classes)
        ax1.bar(range(self.num_classes), true_counts, color='skyblue')
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(self.num_classes))
        ax1.set_xticklabels(self.class_names, rotation=45)
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred, minlength=self.num_classes)
        ax2.bar(range(self.num_classes), pred_counts, color='lightcoral')
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(self.num_classes))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_results(self, metrics, y_true, y_pred, y_prob, results_dir):
        """
        Save evaluation results to files
        """
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(results_dir, f'metrics_{timestamp}.csv'), index=False)
        
        # Save detailed classification report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df.to_csv(os.path.join(results_dir, f'classification_report_{timestamp}.csv'))
        
        # Plot and save confusion matrix
        cm_path = os.path.join(results_dir, f'confusion_matrix_{timestamp}.png')
        self.plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
        
        # Plot and save ROC curves
        if self.num_classes > 2:
            roc_path = os.path.join(results_dir, f'roc_curves_{timestamp}.png')
            self.plot_roc_curves(y_true, y_prob, roc_path)
            
            # Plot and save PR curves
            pr_path = os.path.join(results_dir, f'pr_curves_{timestamp}.png')
            self.plot_precision_recall_curves(y_true, y_prob, pr_path)
        
        # Plot and save class distribution
        dist_path = os.path.join(results_dir, f'class_distribution_{timestamp}.png')
        self.plot_class_distribution(y_true, y_pred, dist_path)
        
        print(f"Evaluation results saved to {results_dir}")
    
    def print_metrics_summary(self, metrics):
        """
        Print a summary of evaluation metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nOverall Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            print(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            precision = metrics.get(f'precision_{class_name}', 0)
            recall = metrics.get(f'recall_{class_name}', 0)
            f1 = metrics.get(f'f1_{class_name}', 0)
            print(f"{class_name:15} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        print("\n" + "="*50)

def evaluate_saved_model(model_path, test_loader, device, class_names=None):
    """
    Evaluate a saved model
    """
    from models.cnn_model import create_model
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create and load model
    model = create_model(config['model_type'], config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, class_names)
    metrics, y_true, y_pred, y_prob = evaluator.evaluate_model(test_loader)
    evaluator.print_metrics_summary(metrics)
    
    return metrics, y_true, y_pred, y_prob

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """
        Generate Class Activation Map
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=[1, 2])
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = torch.relu(cam)
        cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()

if __name__ == "__main__":
    print("Evaluation utilities loaded successfully!")
    print("Available classes:")
    print("- ModelEvaluator: Comprehensive model evaluation")
    print("- GradCAM: Model interpretability with Grad-CAM")
    print("- evaluate_saved_model: Evaluate a saved model checkpoint")