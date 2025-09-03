import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from data_preprocessing import prepare_data
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

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
        self.softmax = nn.Softmax(dim=1)
        
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

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """
    Train the LSTM model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
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
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, label_encoder, device='cuda'):
    """
    Evaluate the model and print detailed results
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Convert back to emoji labels
    predicted_emojis = label_encoder.inverse_transform(all_predictions)
    true_emojis = label_encoder.inverse_transform(all_targets)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_emojis, predicted_emojis))
    
    return accuracy, all_predictions, all_targets

def plot_training_history(train_losses, train_accuracies, test_accuracies):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, preprocessing_data, model_path='emoji_lstm_model.pth'):
    """
    Save the trained model and preprocessing data
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(preprocessing_data['vocab_to_idx']),
            'embedding_dim': model.embedding.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_classes': len(preprocessing_data['label_encoder'].classes_),
            'num_layers': model.num_layers
        },
        'preprocessing_data': preprocessing_data
    }, model_path)
    
    print(f"Model saved to {model_path}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Prepare data
    print("Preparing data...")
    train_dataset, test_dataset, preprocessing_data = prepare_data('sentence_emoji_dataset.csv')
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    vocab_size = len(preprocessing_data['vocab_to_idx'])
    embedding_dim = 128
    hidden_dim = 256
    num_classes = len(preprocessing_data['label_encoder'].classes_)
    num_layers = 2
    dropout = 0.3
    
    print(f"\nModel configuration:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # Create model
    model = EmojiLSTM(vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout)
    
    # Train model
    print("\nStarting training...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, device=device
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, predictions, targets = evaluate_model(
        model, test_loader, preprocessing_data['label_encoder'], device
    )
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, test_accuracies)
    
    # Save model
    save_model(model, preprocessing_data)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
