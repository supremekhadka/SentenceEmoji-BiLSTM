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
from sklearn.model_selection import train_test_split  # For stratified split
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
        self.embedding_dropout = nn.Dropout(0.4)  
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0,  # LSTM internal dropout
            bidirectional=True
        )
        
        # Dropout after LSTM
        self.lstm_dropout = nn.Dropout(dropout)  # After LSTM
        
        # Single dense layer for final prediction
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional, direct to output
        
    def forward(self, x):
        # Embedding with light dropout
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last output from both directions
        # For bidirectional LSTM, concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply dropout after LSTM
        hidden = self.lstm_dropout(hidden)
        
        # Single dense layer for final prediction
        out = self.fc(hidden)
        
        return out

def train_model_gpu(model, train_loader, test_loader, num_epochs=30, learning_rate=0.001, use_early_stopping=True):
    """
    GPU-optimized training function
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        use_early_stopping: Whether to use early stopping (default: True)
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Increased weight decay
    
    # More aggressive learning rate scheduling to prevent overfitting
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []  # Track validation loss for learning rate scheduling
    
    # Early stopping parameters
    best_test_acc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    
    print(f"🔥 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"🧪 Test batches: {len(test_loader)}")
    print(f"⏱️ Early stopping: {'Enabled' if use_early_stopping else 'Disabled'}")
    if use_early_stopping:
        print(f"   Patience: {early_stop_patience} epochs")
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
        total_test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                test_loss = criterion(output, target)
                total_test_loss += test_loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct_test / total_test
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        test_losses.append(avg_test_loss)
        
        # Learning rate scheduling based on validation loss
        scheduler.step(avg_test_loss)
        
        # Early stopping check (only if enabled)
        if use_early_stopping:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1
        else:
            # If early stopping is disabled, just track best accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'🚀 Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)')
        print(f'   Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'   Test Loss:  {avg_test_loss:.4f}, Test Acc:  {test_acc:.2f}%')
        print(f'   Best Test Acc: {best_test_acc:.2f}% | LR: {current_lr:.2e}')
        print(f'   GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        if use_early_stopping:
            print(f'   Patience: {patience_counter}/{early_stop_patience}')
        print("-" * 60)
        
        # Early stopping check (only if enabled)
        if use_early_stopping and patience_counter >= early_stop_patience:
            print(f"⏹️ Early stopping triggered! Best test accuracy: {best_test_acc:.2f}%")
            # Load best model
            model.load_state_dict(torch.load('best_model_temp.pth'))
            break
        
        # Clear GPU cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"🎉 Training completed in {total_time/60:.1f} minutes")
    print(f"📊 Best test accuracy achieved: {best_test_acc:.2f}%")
    
    # Cleanup temporary file (only if early stopping was used)
    if use_early_stopping and os.path.exists('best_model_temp.pth'):
        os.remove('best_model_temp.pth')
    
    return train_losses, train_accuracies, test_accuracies

def main():
    # Check CUDA availability but don't force exit
    print(f"🔍 Checking CUDA availability...")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available! Falling back to CPU training.")
        print("   This will be significantly slower.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print("🚀 GPU TRAINING MODE - TWITTER DATASET")
    print("="*60)
    
    # Prepare data with comprehensive cleaning
    print("📂 Loading and cleaning emoji dataset...")
    if not os.path.exists('emoji_dataset.csv'):
        print("❌ Emoji dataset not found! Please ensure you have the dataset file.")
        return
    
    # Load and preprocess data but don't split yet
    import pandas as pd
    from data_preprocessing import DataCleaner, build_vocabulary, EmojiDataset
    from sklearn.preprocessing import LabelEncoder
    
    # Load raw data
    df = pd.read_csv('emoji_dataset.csv')
    print(f"📊 Original dataset: {len(df)} samples")
    
    # Clean data
    cleaner = DataCleaner()
    df['cleaned_text'] = df['sentence'].apply(cleaner.clean_tweet)  # Use correct method name
    
    # Filter valid texts
    valid_mask = df['cleaned_text'].apply(lambda x: len(x.split()) >= 2)
    df_clean = df[valid_mask].copy()
    print(f"📊 After cleaning: {len(df_clean)} samples")
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['cleaned_text'])
    print(f"📊 After deduplication: {len(df_clean)} samples")
    
    # Remove samples with corrupted emoji "�"
    before_emoji_filter = len(df_clean)
    df_clean = df_clean[df_clean['emoji'] != '�'].copy()
    removed_corrupted = before_emoji_filter - len(df_clean)
    print(f"📊 After removing corrupted emoji '�': {len(df_clean)} samples (removed {removed_corrupted} samples)")
    
    # Limit samples if needed
    if len(df_clean) > 50000:
        # Stratified sampling to maintain class distribution
        df_clean = df_clean.groupby('emoji').apply(lambda x: x.sample(min(len(x), 50000 // df_clean['emoji'].nunique()))).reset_index(drop=True)
        print(f"📊 After stratified sampling: {len(df_clean)} samples")
    
    # Build vocabulary
    vocab_to_idx, idx_to_vocab = build_vocabulary(df_clean['cleaned_text'].tolist(), min_freq=3)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df_clean['emoji'])
    
    print(f"📊 Vocabulary size: {len(vocab_to_idx)}")
    print(f"📊 Number of emoji classes: {len(label_encoder.classes_)}")
    print(f"📊 Class distribution:")
    unique, counts = np.unique(encoded_labels, return_counts=True)
    for emoji, count in zip(label_encoder.classes_, counts):
        print(f"   {emoji}: {count} samples ({count/len(encoded_labels)*100:.1f}%)")
    
    # STRATIFIED train-test split to ensure balanced classes
    sentences = df_clean['cleaned_text'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        sentences, 
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels  # This ensures balanced class distribution
    )
    
    print(f"📊 Stratified split completed:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Verify class distribution in splits
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print(f"📊 Training class distribution:")
    for i, (emoji, count) in enumerate(zip(label_encoder.classes_, train_counts)):
        print(f"   {emoji}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Create datasets
    max_length = 30
    train_dataset = EmojiDataset(X_train, y_train, vocab_to_idx, max_length)
    test_dataset = EmojiDataset(X_test, y_test, vocab_to_idx, max_length)
    
    # Store preprocessing data
    preprocessing_data = {
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'label_encoder': label_encoder,
        'max_length': max_length
    }
    
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
    hidden_dim = 128     # Increased for more capacity
    num_classes = len(preprocessing_data['label_encoder'].classes_)
    num_layers = 1       # Increased depth
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
    
    # Training configuration
    use_early_stopping = True  # Set to False to disable early stopping
    num_epochs = 50  # More epochs if no early stopping
    
    print(f"🔧 TRAINING CONFIGURATION:")
    print(f"   Early stopping: {'Enabled' if use_early_stopping else 'Disabled'}")
    print(f"   Max epochs: {num_epochs}")
    
    # Train model on GPU
    print("\n🔥 STARTING GPU TRAINING...")
    train_losses, train_accuracies, test_accuracies = train_model_gpu(
        model, train_loader, test_loader, 
        num_epochs=num_epochs, 
        learning_rate=1e-4,
        use_early_stopping=use_early_stopping
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
