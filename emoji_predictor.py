import torch
import torch.nn as nn
import pickle
import re
import numpy as np
from gpu_training import EmojiLSTM  # Import from gpu_training.py which has the enhanced architecture

class EmojiPredictor:
    def __init__(self, model_path='emoji_lstm_model.pth'):
        """
        Initialize the emoji predictor with the trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.preprocessing_data = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model and preprocessing data
        """
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Load preprocessing data
            self.preprocessing_data = checkpoint['preprocessing_data']
            
            # Load model configuration
            model_config = checkpoint['model_config']
            
            # Create model instance
            self.model = EmojiLSTM(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_classes=model_config['num_classes'],
                num_layers=model_config['num_layers'],
                dropout=0.4  # Add the dropout parameter used during training
            )
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Using device: {self.device}")
            print(f"Vocabulary size: {len(self.preprocessing_data['vocab_to_idx'])}")
            print(f"Number of emoji classes: {len(self.preprocessing_data['label_encoder'].classes_)}")
            
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found. Please train the model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def clean_text(self, text):
        """
        Apply the same cleaning used during training
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and links
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'(?:(?:www\.)|(?:http://)|(?:https://))?\w+\.\w+(?:\.\w+)*(?:/\S*)?', '', text)
        text = re.sub(r'pic\.twitter\.com/\w+', '', text)
        
        # Remove mentions and RT
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'^RT\s+', '', text)
        text = re.sub(r'\s+via\s+@\w+', '', text, flags=re.IGNORECASE)
        
        # Remove hashtags (keep text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emoji text representations
        text = re.sub(r':\w+:', '', text)
        
        # Reduce repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def preprocess_sentence(self, sentence):
        """
        Preprocess a sentence for prediction using comprehensive cleaning
        """
        # Clean the text first
        cleaned_text = self.clean_text(sentence)
        
        if not cleaned_text:
            return torch.zeros(1, self.preprocessing_data['max_length'], dtype=torch.long).to(self.device)
        
        # Tokenize
        tokens = cleaned_text.split()
        
        # Convert tokens to indices
        vocab_to_idx = self.preprocessing_data['vocab_to_idx']
        max_length = self.preprocessing_data['max_length']
        
        indices = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([vocab_to_idx['<PAD>']] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        return torch.tensor([indices], dtype=torch.long).to(self.device)
    
    def predict_emoji(self, sentence, top_k=3):
        """
        Predict emoji(s) for a given sentence
        
        Args:
            sentence (str): Input sentence
            top_k (int): Number of top predictions to return
        
        Returns:
            list: List of tuples (emoji, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Preprocess the sentence
        input_tensor = self.preprocess_sentence(sentence)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.preprocessing_data['label_encoder'].classes_)), dim=1)
            
            # Convert to emoji labels
            label_encoder = self.preprocessing_data['label_encoder']
            predictions = []
            
            for i in range(top_probs.shape[1]):
                emoji_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                emoji = label_encoder.inverse_transform([emoji_idx])[0]
                predictions.append((emoji, confidence))
        
        return predictions
    
    def predict_batch(self, sentences, top_k=3):
        """
        Predict emojis for a batch of sentences
        """
        predictions = []
        for sentence in sentences:
            pred = self.predict_emoji(sentence, top_k)
            predictions.append(pred)
        return predictions
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if self.preprocessing_data is None:
            return "No model loaded"
        
        info = {
            'model_type': 'LSTM-RNN Emoji Predictor',
            'vocabulary_size': len(self.preprocessing_data['vocab_to_idx']),
            'max_sequence_length': self.preprocessing_data['max_length'],
            'emoji_classes': list(self.preprocessing_data['label_encoder'].classes_),
            'num_classes': len(self.preprocessing_data['label_encoder'].classes_),
            'device': str(self.device),
            'data_source': 'Social media text with comprehensive cleaning'
        }
        
        return info

def test_predictor():
    """
    Test function for the emoji predictor
    """
    try:
        # Initialize predictor
        predictor = EmojiPredictor()
        
        # Test sentences with social media-like content
        test_sentences = [
            "I love this beautiful day so much",
            "This is absolutely amazing and incredible",
            "I am feeling really sad today",
            "Happy birthday to you my friend",
            "Good night and sweet dreams everyone",
            "This weather is perfect for a picnic",
            "Can't believe this happened to me",
            "So excited for the weekend",
            "Feeling grateful for everything",
            "This food tastes incredible"
        ]
        
        print("Testing Emoji Predictor...")
        print("="*60)
        
        # Model info
        model_info = predictor.get_model_info()
        print(f"Model Info:")
        print(f"- Model Type: {model_info['model_type']}")
        print(f"- Vocabulary Size: {model_info['vocabulary_size']:,}")
        print(f"- Max Sequence Length: {model_info['max_sequence_length']}")
        print(f"- Number of Emoji Classes: {model_info['num_classes']}")
        print(f"- Data Source: {model_info['data_source']}")
        print(f"- Device: {model_info['device']}")
        print()
        
        # Test predictions
        for i, sentence in enumerate(test_sentences, 1):
            predictions = predictor.predict_emoji(sentence, top_k=3)
            cleaned = predictor.clean_text(sentence)
            
            print(f"{i:2d}. Original: '{sentence}'")
            print(f"    Cleaned:  '{cleaned}'")
            print("    Top 3 predictions:")
            for j, (emoji, confidence) in enumerate(predictions, 1):
                print(f"      {j}. {emoji} ({confidence:.3f})")
            print()
    
    except Exception as e:
        print(f"Error testing predictor: {e}")
        print("Make sure you have trained the model first by running gpu_training.py")

if __name__ == "__main__":
    test_predictor()
