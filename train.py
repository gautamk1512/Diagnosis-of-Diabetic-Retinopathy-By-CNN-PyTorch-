import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import os
import time
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Import custom modules
from models.cnn_model import create_model, count_parameters
from utils.data_preprocessing import (
    load_dataset_from_csv, split_dataset, create_data_loaders, 
    calculate_class_weights
)

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer:
    """
    Training class for Diabetic Retinopathy Classification
    """
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Setup loss function
        if config['loss_function'] == 'focal':
            self.criterion = FocalLoss(alpha=config.get('focal_alpha', 1), 
                                     gamma=config.get('focal_gamma', 2))
        elif config['loss_function'] == 'weighted_ce':
            class_weights = config.get('class_weights', None)
            if class_weights is not None:
                class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), 
                                      lr=config['learning_rate'],
                                      weight_decay=config.get('weight_decay', 1e-4))
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(model.parameters(),
                                     lr=config['learning_rate'],
                                     momentum=config.get('momentum', 0.9),
                                     weight_decay=config.get('weight_decay', 1e-4))
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(),
                                       lr=config['learning_rate'],
                                       weight_decay=config.get('weight_decay', 1e-2))
        
        # Setup scheduler
        if config.get('scheduler') == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                           factor=0.5, patience=5)
        elif config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, 
                                             T_max=config['num_epochs'])
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.get('patience', 10))
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """
        Setup logging for training
        """
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config['grad_clip'])
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                self.logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                               f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """
        Validate for one epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self):
        """
        Main training loop
        """
        self.logger.info(f"Starting training with {count_parameters(self.model):,} parameters")
        self.logger.info(f"Training on {self.device}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_predictions, val_targets = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f'Epoch {epoch+1}/{self.config["num_epochs"]} '
                f'({epoch_time:.2f}s) - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best_model.pth')
                self.logger.info(f'New best validation accuracy: {val_acc:.2f}%')
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f'Early stopping at epoch {epoch+1}')
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc, f'checkpoint_epoch_{epoch+1}.pth')
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time/60:.2f} minutes')
        self.logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, filename):
        """
        Save model checkpoint
        """
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    
    def plot_training_history(self):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(results_dir, f'training_history_{timestamp}.png'))
        plt.show()

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

if __name__ == "__main__":
    # Example training configuration
    config = {
        'model_type': 'resnet50',
        'num_classes': 5,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'scheduler': 'reduce_on_plateau',
        'loss_function': 'weighted_ce',
        'weight_decay': 1e-4,
        'patience': 10,
        'grad_clip': 1.0,
        'image_size': (224, 224)
    }
    
    print("Training script loaded successfully!")
    print("Available classes:")
    print("- Trainer: Main training class")
    print("- EarlyStopping: Early stopping implementation")
    print("- FocalLoss: Focal loss for class imbalance")
    print("- load_checkpoint: Load saved model")
    print("\nTo start training, create a Trainer instance and call train()")