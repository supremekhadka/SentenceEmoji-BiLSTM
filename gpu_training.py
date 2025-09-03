"""
GPU-Optimized Model Training Script with Comprehensive Data Cleaning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from data_preprocessing import prepare_data  # Use cleaned data
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time
import oszed Model Training Script for Twitter Dataset with Comprehensive Data Cleaning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from twitter_data_preprocessing import prepare_twitter_data  # Use cleaned Twitter data
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time
import os

class EmojiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(EmojiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last output from both directions
        # For bidirectional LSTM, concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layers
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_model_gpu(model, train_loader, test_loader, num_epochs=30, learning_rate=0.001):
    """
    GPU-optimized training function
    """
    # Force GPU usage
    device = torch.device('cuda')
    print(f"🚀 FORCED GPU TRAINING")
    print(f"Device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"GPU Memory Available: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    
    # Move model to GPU
    model = model.to(device)
    
    # Ensure all model parameters are on GPU
    for param in model.parameters():
        if not param.is_cuda:
            print("⚠️ Warning: Parameter not on GPU!")
            param.data = param.data.cuda()
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.8)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"🔥 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"🧪 Test batches: {len(test_loader)}")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Ensure data is on GPU
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Print batch progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct_test / total_test
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f'🚀 Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)')
        print(f'   Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        print(f'   GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        print(f'   Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print("-" * 60)
        
        # Clear GPU cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"🎉 Training completed in {total_time/60:.1f} minutes")
    
    return train_losses, train_accuracies, test_accuracies

def main():
    # Force CUDA device
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please check your PyTorch installation.")
        return
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    print("🚀 GPU TRAINING MODE - TWITTER DATASET")
    print("="*60)
    
    # Prepare data with comprehensive cleaning
    print("📂 Loading and cleaning emoji dataset...")
    if not os.path.exists('emoji_dataset.csv'):
        print("❌ Emoji dataset not found! Please ensure you have the dataset file.")
        return
        
    train_dataset, test_dataset, preprocessing_data = prepare_data(
        'emoji_dataset.csv',
        max_samples=50000,  # Large enough for good training
        max_length=30,      # Longer for social media content
        min_freq=3          # Filter rare words
    )
    
    # Create data loaders with GPU optimization
    batch_size = 64  # Increased batch size for GPU
    num_workers = 4   # Parallel data loading
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # GPU optimization
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # GPU optimization
    )
    
    # Model parameters - optimized for larger dataset
    vocab_size = len(preprocessing_data['vocab_to_idx'])
    embedding_dim = 256  # Increased for better representations
    hidden_dim = 512     # Increased for more capacity
    num_classes = len(preprocessing_data['label_encoder'].classes_)
    num_layers = 3       # Increased depth
    dropout = 0.4        # Slightly higher dropout
    
    print(f"📊 MODEL CONFIGURATION:")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Number of emoji classes: {num_classes}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   LSTM layers: {num_layers}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    
    # Create model
    model = EmojiLSTM(vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout)
    
    # Train model on GPU
    print("\n🔥 STARTING GPU TRAINING...")
    train_losses, train_accuracies, test_accuracies = train_model_gpu(
        model, train_loader, test_loader, num_epochs=25, learning_rate=0.001
    )
    
    # Save model
    print("\n💾 Saving trained model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'num_layers': num_layers
        },
        'preprocessing_data': preprocessing_data,
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    }, 'emoji_lstm_model.pth')
    
    print("✅ Model saved as 'emoji_lstm_model.pth'")
    print("🎉 GPU TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
