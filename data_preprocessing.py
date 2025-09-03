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

class DataCleaner:
    def __init__(self):
        """Initialize data cleaner with comprehensive patterns"""
        
        # Compiled regex patterns for better performance
        self.patterns = {
            'mentions': re.compile(r'@\w+'),                    # @username
            'hashtags': re.compile(r'#\w+'),                    # #hashtag (optional: keep or remove)
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'short_urls': re.compile(r'(?:(?:www\.)|(?:http://)|(?:https://))?\w+\.\w+(?:\.\w+)*(?:/\S*)?'),
            'pic_twitter': re.compile(r'pic\.twitter\.com/\w+'), # Picture links
            'rt': re.compile(r'^RT\s+'),                        # Retweet indicator
            'via': re.compile(r'\s+via\s+@\w+', re.IGNORECASE), # Via mentions
            'numbers': re.compile(r'\b\d+\b'),                  # Standalone numbers
            'extra_spaces': re.compile(r'\s+'),                 # Multiple spaces
            'special_chars': re.compile(r'[^\w\s]'),            # Special characters except spaces
            'repeated_chars': re.compile(r'(.)\1{2,}'),         # Repeated characters (e.g., "sooooo" -> "so")
            'emoji_text': re.compile(r':\w+:'),                 # Emoji text like :smile:
        }
    
    def clean_tweet(self, text):
        """
        Comprehensive tweet cleaning
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase first
        text = text.lower()
        
        # Remove RT indicator
        text = self.patterns['rt'].sub('', text)
        
        # Remove mentions
        text = self.patterns['mentions'].sub('', text)
        
        # Remove URLs and links
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['short_urls'].sub('', text)
        text = self.patterns['pic_twitter'].sub('', text)
        
        # Remove hashtags (keeping the text after #)
        text = self.patterns['hashtags'].sub(lambda m: m.group(0)[1:], text)
        
        # Remove via mentions
        text = self.patterns['via'].sub('', text)
        
        # Remove emoji text representations
        text = self.patterns['emoji_text'].sub('', text)
        
        # Reduce repeated characters (e.g., "loooove" -> "love")
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Remove standalone numbers
        text = self.patterns['numbers'].sub('', text)
        
        # Remove special characters and punctuation
        text = self.patterns['special_chars'].sub(' ', text)
        
        # Clean up extra spaces
        text = self.patterns['extra_spaces'].sub(' ', text)
        
        # Strip leading/trailing spaces
        text = text.strip()
        
        return text
    
    def is_valid_tweet(self, text):
        """
        Check if tweet is valid after cleaning
        """
        if not text or len(text) < 3:
            return False
        
        # Check if text has at least 2 words
        words = text.split()
        if len(words) < 2:
            return False
        
        # Check if text is not just spaces or common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 1]
        
        if len(meaningful_words) < 1:
            return False
        
        return True

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
        # Simple tokenization - text is already cleaned
        tokens = text.split()
        return tokens

def clean_and_preprocess_twitter_data(csv_path, max_samples=None):
    """
    Load and clean Twitter emoji dataset with comprehensive preprocessing
    """
    print("Loading Twitter emoji dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display sample data before cleaning
    print("\nSample data before cleaning:")
    print(df.head())
    
    # Initialize cleaner
    cleaner = TwitterDataCleaner()
    
    # Clean the text data
    print("\nCleaning tweet text...")
    df['cleaned_text'] = df['sentence'].apply(cleaner.clean_tweet)  # Use 'sentence' column
    
    # Filter valid tweets
    print("Filtering valid tweets...")
    valid_mask = df['cleaned_text'].apply(cleaner.is_valid_tweet)
    df_clean = df[valid_mask].copy()
    
    print(f"After text cleaning and validation: {len(df_clean)} tweets (removed {len(df) - len(df_clean)} invalid tweets)")
    
    # Remove duplicates based on cleaned text
    print("Removing duplicates...")
    df_clean = df_clean.drop_duplicates(subset=['cleaned_text'], keep='first')
    print(f"After removing duplicates: {len(df_clean)} tweets")
    
    # Limit samples if specified
    if max_samples and len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=42)
        print(f"Sampled to {max_samples} tweets for training")
    
    # Display sample cleaned data
    print("\nSample cleaned data:")
    for i in range(min(5, len(df_clean))):
        original = df.iloc[i]['sentence'] if i < len(df) else "N/A"  # Use 'sentence' column
        cleaned = df_clean.iloc[i]['cleaned_text']
        emoji = df_clean.iloc[i]['emoji']  # Use 'emoji' column
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}")
        print(f"Emoji:    {emoji}")
        print("-" * 50)
    
    # Emoji distribution
    print(f"\nEmoji distribution (top 10):")
    emoji_counts = df_clean['emoji'].value_counts()  # Use 'emoji' column
    print(emoji_counts.head(10))
    print(f"Total unique emojis: {len(emoji_counts)}")
    
    # Text statistics
    word_counts = df_clean['cleaned_text'].apply(lambda x: len(x.split()))
    print(f"\nText statistics:")
    print(f"Average words per tweet: {word_counts.mean():.2f}")
    print(f"Min words: {word_counts.min()}")
    print(f"Max words: {word_counts.max()}")
    print(f"Median words: {word_counts.median():.2f}")
    
    return df_clean

def build_vocabulary(sentences, min_freq=2):
    """
    Build vocabulary from cleaned sentences
    """
    print("Building vocabulary...")
    
    # Tokenize all sentences
    all_tokens = []
    for sentence in sentences:
        tokens = sentence.split()  # Already cleaned, just split
        all_tokens.extend(tokens)
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    print(f"Total tokens before filtering: {len(token_counts)}")
    
    # Filter by minimum frequency
    vocab = [token for token, count in token_counts.items() if count >= min_freq]
    
    # Add special tokens
    vocab = ['<PAD>', '<UNK>'] + vocab
    
    # Create vocabulary mappings
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_vocab = {idx: token for idx, token in enumerate(vocab)}
    
    print(f"Vocabulary size after filtering (min_freq={min_freq}): {len(vocab)}")
    print(f"Most common tokens: {list(token_counts.most_common(20))}")
    
    return vocab_to_idx, idx_to_vocab

def prepare_data(csv_path, test_size=0.2, max_length=30, max_samples=50000, min_freq=3):
    """
    Complete Twitter data preparation pipeline
    """
    # Clean and load data
    df = clean_and_preprocess_twitter_data(csv_path, max_samples)
    
    # Extract sentences and emojis
    sentences = df['cleaned_text'].tolist()
    emojis = df['emoji'].tolist()  # Use 'emoji' column
    
    # Filter emojis that appear frequently enough
    emoji_counts = Counter(emojis)
    frequent_emojis = [emoji for emoji, count in emoji_counts.items() if count >= 20]
    print(f"\nFiltering to emojis with at least 20 occurrences: {len(frequent_emojis)} emojis")
    
    # Filter data to only include frequent emojis
    mask = [emoji in frequent_emojis for emoji in emojis]
    sentences = [sentences[i] for i in range(len(sentences)) if mask[i]]
    emojis = [emojis[i] for i in range(len(emojis)) if mask[i]]
    
    print(f"Final dataset size: {len(sentences)} tweets")
    
    # Encode emojis
    label_encoder = LabelEncoder()
    encoded_emojis = label_encoder.fit_transform(emojis)
    
    print(f"\nEmoji encoding (final classes):")
    for i, emoji in enumerate(label_encoder.classes_):
        count = emoji_counts[emoji]
        print(f"{emoji}: {i} (count: {count})")
    
    # Build vocabulary
    vocab_to_idx, idx_to_vocab = build_vocabulary(sentences, min_freq)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, encoded_emojis, test_size=test_size, random_state=42, stratify=encoded_emojis
        )
    except ValueError:
        # Fallback to non-stratified if some classes have too few samples
        print("Warning: Using non-stratified split due to class imbalance")
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, encoded_emojis, test_size=test_size, random_state=42
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
    
    with open('twitter_preprocessing_data.pkl', 'wb') as f:
        pickle.dump(preprocessing_data, f)
    
    print("\nPreprocessing data saved to 'twitter_preprocessing_data.pkl'")
    
    return train_dataset, test_dataset, preprocessing_data

if __name__ == "__main__":
    # Check if emoji dataset exists
    emoji_csv = "emoji_dataset.csv"
    if not os.path.exists(emoji_csv):
        print(f"Error: {emoji_csv} not found!")
        print("Please ensure you have the emoji dataset file.")
        exit(1)
    
    # Prepare data with comprehensive cleaning
    print("="*60)
    print("EMOJI DATASET - COMPREHENSIVE PREPROCESSING")
    print("="*60)
    
    train_dataset, test_dataset, preprocessing_data = prepare_data(
        emoji_csv, 
        max_samples=50000,  # Limit for reasonable training time
        max_length=30,      # Longer sequences for better context
        min_freq=3          # More restrictive vocabulary
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE! Ready for training.")
    print("Run: python gpu_training.py")
    print("="*60)
