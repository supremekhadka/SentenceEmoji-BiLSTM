import torch
import torch.nn as nn
import pickle
import re
import numpy as np
from model_training import EmojiLSTM

class EmojiPredictor:
    def __init__(self, model_path='emoji_lstm_model.pth'):
        """
        Initialize the emoji predictor with a trained model
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
                num_layers=model_config['num_layers']
            )
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Using device: {self.device}")
            
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found. Please train the model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_sentence(self, sentence):
        """
        Preprocess a single sentence for prediction
        """
        # Tokenize the sentence
        text = sentence.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()
        
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
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            # Convert to emoji labels
            label_encoder = self.preprocessing_data['label_encoder']
            predictions = []
            
            for i in range(top_k):
                emoji_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                emoji = label_encoder.inverse_transform([emoji_idx])[0]
                predictions.append((emoji, confidence))
        
        return predictions
    
    def predict_batch(self, sentences, top_k=3):
        """
        Predict emojis for a batch of sentences
        
        Args:
            sentences (list): List of input sentences
            top_k (int): Number of top predictions to return for each sentence
        
        Returns:
            list: List of predictions for each sentence
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
            'vocabulary_size': len(self.preprocessing_data['vocab_to_idx']),
            'max_sequence_length': self.preprocessing_data['max_length'],
            'emoji_classes': list(self.preprocessing_data['label_encoder'].classes_),
            'num_classes': len(self.preprocessing_data['label_encoder'].classes_),
            'device': str(self.device)
        }
        
        return info

def test_predictor():
    """
    Test function for the emoji predictor
    """
    try:
        # Initialize predictor
        predictor = EmojiPredictor()
        
        # Test sentences
        test_sentences = [
            "I love you so much",
            "This is amazing",
            "I am very sad today",
            "Happy birthday to you",
            "Good night sweet dreams"
        ]
        
        print("Testing Emoji Predictor...")
        print("="*50)
        
        # Model info
        model_info = predictor.get_model_info()
        print(f"Model Info:")
        print(f"- Vocabulary Size: {model_info['vocabulary_size']}")
        print(f"- Max Sequence Length: {model_info['max_sequence_length']}")
        print(f"- Number of Emoji Classes: {model_info['num_classes']}")
        print(f"- Available Emojis: {', '.join(model_info['emoji_classes'])}")
        print(f"- Device: {model_info['device']}")
        print()
        
        # Test predictions
        for sentence in test_sentences:
            predictions = predictor.predict_emoji(sentence, top_k=3)
            print(f"Sentence: '{sentence}'")
            print("Top 3 predictions:")
            for i, (emoji, confidence) in enumerate(predictions, 1):
                print(f"  {i}. {emoji} ({confidence:.3f})")
            print()
    
    except Exception as e:
        print(f"Error testing predictor: {e}")
        print("Make sure you have trained the model first by running model_training.py")

if __name__ == "__main__":
    test_predictor()
