# ===============================
# fixed_train_disease_model.py
# ===============================
# Enhanced GPU-ready training of MobileNetV2 for Plant Disease Detection
# Dataset: PlantVillage (organized by folders)
# FIXED VERSION - Addresses overfitting issues

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =====================
# FIXED Configuration Class
# =====================
class Config:
    # Paths
    DATA_DIR = r"C:\Users\Godwin Arulraj\Desktop\sih2025\data\PlantVillage"
    MODEL_DIR = r"C:\Users\Godwin Arulraj\Desktop\sih2025\models"
    MODEL_PATH = os.path.join(MODEL_DIR, "disease_model_fixed.pth")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "disease_model_best_fixed.pth")
    HISTORY_PATH = os.path.join(MODEL_DIR, "training_history_fixed.json")
    
    # FIXED Hyperparameters - Reduced overfitting
    IMG_SIZE = 224
    BATCH_SIZE = 16          # Reduced from 32 for more regularization
    EPOCHS = 25              # Increased epochs since we have slower learning
    LR = 5e-5               # Reduced from 1e-4 (5x smaller)
    TRAIN_SPLIT = 0.8
    
    # FIXED Training parameters - More aggressive regularization
    DROPOUT_RATE = 0.6       # Increased from 0.3
    WEIGHT_DECAY = 0.1       # Increased regularization
    PATIENCE = 5             # Reduced from 7 (stop sooner)
    MIN_DELTA = 0.001
    SCHEDULER_PATIENCE = 2   # Reduced from 3 (more aggressive LR reduction)
    SCHEDULER_FACTOR = 0.3   # Reduced from 0.5 (more aggressive reduction)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# =====================
# Utility Classes (Unchanged)
# =====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class TrainingLogger:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epochs': []
        }
    
    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rates'].append(lr)
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

# =====================
# ENHANCED Data Preparation - More Aggressive Augmentation
# =====================
def get_data_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomRotation(45),                    # Increased from 30
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),            # Increased from 0.2
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.6, 1.0)),  # More aggressive crop
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Enhanced
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),     # NEW: Affine
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),  # NEW: Blur
        transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.2),             # NEW: Posterize
        transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.2),       # NEW: Solarize
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # NEW: Random erasing (additional regularization)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_data_loaders():
    train_transforms, val_transforms = get_data_transforms()
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(config.DATA_DIR, transform=train_transforms)
    
    # Split dataset
    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use manual seed for reproducible splits
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transforms
    
    # Create data loaders with reduced batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes

# =====================
# IMPROVED Model Creation - Better Regularization
# =====================
def create_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    
    # FIXED: Freeze more layers (all but the last block)
    for param in model.features[:-1].parameters():  # Changed from [:-3]
        param.requires_grad = False
    
    # FIXED: Enhanced classifier with more regularization
    model.classifier = nn.Sequential(
        nn.Dropout(config.DROPOUT_RATE),                    # Increased dropout
        nn.Linear(model.last_channel, 256),                 # Smaller hidden layer
        nn.BatchNorm1d(256),                                # NEW: Batch normalization
        nn.ReLU(inplace=True),
        nn.Dropout(config.DROPOUT_RATE),                    # Additional dropout
        nn.Linear(256, 128),                                # NEW: Additional layer
        nn.BatchNorm1d(128),                                # NEW: Batch normalization
        nn.ReLU(inplace=True),
        nn.Dropout(config.DROPOUT_RATE),                    # More dropout
        nn.Linear(128, num_classes)
    )
    
    return model.to(config.DEVICE)

# =====================
# Training Functions with Label Smoothing
# =====================
class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        
        log_prob = torch.log_softmax(pred, dim=1)
        loss = torch.sum(-smooth_target * log_prob, dim=1)
        return loss.mean()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # FIXED: More aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:  # Less frequent logging
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# =====================
# Evaluation Functions (Unchanged)
# =====================
def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    """Generate and save classification report"""
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Generate confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Fixed Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(history['epochs'], history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['epochs'], history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Model Loss - Fixed Version')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['epochs'], history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['epochs'], history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_title('Model Accuracy - Fixed Version')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning Rate
    ax3.plot(history['epochs'], history['learning_rates'], linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Val Accuracy vs Learning Rate
    ax4.scatter(history['learning_rates'], history['val_acc'], alpha=0.6, s=30)
    ax4.set_title('Validation Accuracy vs Learning Rate')
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# =====================
# MAIN Training Function with Fixes
# =====================
def main():
    print(f"🚀 FIXED Plant Disease Detection Training")
    print(f"🔧 Applied Overfitting Fixes")
    print(f"Device: {config.DEVICE}")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    # Print configuration changes
    print(f"📋 Key Changes Applied:")
    print(f"   • Learning Rate: {config.LR} (reduced from 1e-4)")
    print(f"   • Dropout Rate: {config.DROPOUT_RATE} (increased from 0.3)")
    print(f"   • Batch Size: {config.BATCH_SIZE} (reduced from 32)")
    print(f"   • Weight Decay: {config.WEIGHT_DECAY} (increased regularization)")
    print(f"   • Early Stopping Patience: {config.PATIENCE} (reduced from 7)")
    print(f"   • Enhanced data augmentation and label smoothing")
    print(f"   • More frozen layers and batch normalization")
    
    # Prepare data
    train_loader, val_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    
    print(f"\n📊 Dataset Info:")
    print(f"  Classes: {num_classes}")
    print(f"  Class names: {class_names[:5]}...")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(num_classes)
    print(f"\n🏗️ Model: Enhanced MobileNetV2")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # FIXED: Training setup with label smoothing and higher weight decay
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)  # NEW: Label smoothing
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY,  # Increased weight decay
        betas=(0.9, 0.999)
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=config.SCHEDULER_FACTOR,    # More aggressive reduction
        patience=config.SCHEDULER_PATIENCE, # Reduced patience
        verbose=True,
        min_lr=1e-7  # Lower minimum LR
    )
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,          # Reduced patience
        min_delta=config.MIN_DELTA
    )
    logger = TrainingLogger()
    
    # Training loop
    print(f"\n🔥 Starting Fixed Training...")
    best_val_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.log(epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': config.__dict__,
                'fixes_applied': [
                    'Reduced learning rate to 5e-5',
                    'Increased dropout to 0.6',
                    'Enhanced data augmentation',
                    'Added label smoothing',
                    'Increased weight decay',
                    'More frozen layers',
                    'Batch normalization'
                ]
            }, config.BEST_MODEL_PATH)
            print(f"  ⭐ New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if early_stopping(val_acc, model):
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Save final model and history
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    logger.save(config.HISTORY_PATH)
    
    # Generate final evaluation
    print(f"\n📊 Generating Final Evaluation...")
    
    # Load best model for final evaluation
    checkpoint = torch.load(config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, config.DEVICE)
    
    # Generate reports
    report_path = os.path.join(config.MODEL_DIR, "classification_report_fixed.json")
    cm_path = os.path.join(config.MODEL_DIR, "confusion_matrix_fixed.png")
    history_plot_path = os.path.join(config.MODEL_DIR, "training_history_fixed.png")
    
    generate_classification_report(val_labels, val_preds, class_names, report_path)
    plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
    plot_training_history(logger.history, history_plot_path)
    
    print(f"\n✅ Fixed Training Completed!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Final model: {config.MODEL_PATH}")
    print(f"  Best model: {config.BEST_MODEL_PATH}")
    print(f"  Training history: {config.HISTORY_PATH}")
    print(f"  Classification report: {report_path}")
    print(f"  Confusion matrix: {cm_path}")
    print(f"  Training plots: {history_plot_path}")
    
    print(f"\n🎯 Fixes Applied Successfully:")
    fixes = [
        "✓ Reduced learning rate (5x smaller)",
        "✓ Increased dropout rate (2x higher)", 
        "✓ Enhanced data augmentation",
        "✓ Added label smoothing loss",
        "✓ Increased weight decay regularization",
        "✓ More aggressive LR scheduling",
        "✓ Earlier early stopping",
        "✓ Batch normalization layers",
        "✓ More frozen pre-trained layers"
    ]
    for fix in fixes:
        print(f"     {fix}")

if __name__ == "__main__":
    main()