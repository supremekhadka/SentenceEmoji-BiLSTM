import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import string
from collections import Counter
import pickle
import os

class EmojiDataset(Dataset):
    def __init__(self, sentences, emojis, vocab_to_idx, max_length):
        self.sentences = sentences
        self.emojis = emojis
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        emoji = self.emojis[idx]
        
        # Convert sentence to indices
        tokens = self.tokenize(sentence)
        indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(emoji, dtype=torch.long)
    
    def tokenize(self, text):
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()
        return tokens

def clean_and_preprocess_data(csv_path):
    """
    Load and clean the dataset, removing duplicates and preprocessing text
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    # Remove duplicates based on sentence
    print(f"\nBefore removing duplicates: {len(df)} rows")
    df_clean = df.drop_duplicates(subset=['sentence'], keep='first')
    print(f"After removing duplicates: {len(df_clean)} rows")
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df_clean.isnull().sum())
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    print(f"After removing missing values: {len(df_clean)} rows")
    
    # Display emoji distribution
    print(f"\nEmoji distribution:")
    print(df_clean['emoji'].value_counts())
    
    return df_clean

def build_vocabulary(sentences, min_freq=1):
    """
    Build vocabulary from sentences
    """
    print("Building vocabulary...")
    
    # Tokenize all sentences
    all_tokens = []
    for sentence in sentences:
        text = sentence.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()
        all_tokens.extend(tokens)
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Filter by minimum frequency
    vocab = [token for token, count in token_counts.items() if count >= min_freq]
    
    # Add special tokens
    vocab = ['<PAD>', '<UNK>'] + vocab
    
    # Create vocabulary mappings
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_vocab = {idx: token for idx, token in enumerate(vocab)}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Most common tokens: {list(token_counts.most_common(10))}")
    
    return vocab_to_idx, idx_to_vocab

def prepare_data(csv_path, test_size=0.2, max_length=20):
    """
    Complete data preparation pipeline
    """
    # Clean and load data
    df = clean_and_preprocess_data(csv_path)
    
    # Extract sentences and emojis
    sentences = df['sentence'].tolist()
    emojis = df['emoji'].tolist()
    
    # Encode emojis
    label_encoder = LabelEncoder()
    encoded_emojis = label_encoder.fit_transform(emojis)
    
    print(f"\nEmoji encoding:")
    for i, emoji in enumerate(label_encoder.classes_):
        print(f"{emoji}: {i}")
    
    # Build vocabulary
    vocab_to_idx, idx_to_vocab = build_vocabulary(sentences)
    
    # Check if stratified split is possible
    unique, counts = np.unique(encoded_emojis, return_counts=True)
    min_count = min(counts)
    
    if min_count < 2:
        print(f"Warning: Some classes have only {min_count} samples. Using simple random split instead of stratified.")
        # Use simple random split
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, encoded_emojis, test_size=test_size, random_state=42
        )
    else:
        # Use stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, encoded_emojis, test_size=test_size, random_state=42, stratify=encoded_emojis
        )
    
    print(f"\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = EmojiDataset(X_train, y_train, vocab_to_idx, max_length)
    test_dataset = EmojiDataset(X_test, y_test, vocab_to_idx, max_length)
    
    # Save preprocessing objects
    preprocessing_data = {
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'label_encoder': label_encoder,
        'max_length': max_length
    }
    
    with open('preprocessing_data.pkl', 'wb') as f:
        pickle.dump(preprocessing_data, f)
    
    print("\nPreprocessing data saved to 'preprocessing_data.pkl'")
    
    return train_dataset, test_dataset, preprocessing_data

if __name__ == "__main__":
    # Prepare data
    train_dataset, test_dataset, preprocessing_data = prepare_data('sentence_emoji_dataset.csv')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"\nDataLoader created:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nSample batch shape:")
        print(f"Input shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Input sample: {data[0]}")
        print(f"Target sample: {target[0]}")
        break
