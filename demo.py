"""
Emoji Predictor Demo Script
This script demonstrates the complete workflow of the emoji prediction system.
"""

import os
import sys
import torch
from emoji_predictor import EmojiPredictor

def check_setup():
    """Check if all necessary files exist"""
    required_files = [
        'sentence_emoji_dataset.csv',
        'emoji_lstm_model.pth',
        'preprocessing_data.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run the following commands in order:")
        print("1. python data_preprocessing.py")
        print("2. python model_training.py")
        return False
    
    print("✅ All required files found!")
    return True

def demo_predictions():
    """Demonstrate emoji predictions"""
    print("\n" + "="*60)
    print("🚀 EMOJI PREDICTOR DEMONSTRATION")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = EmojiPredictor()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Display model info
    model_info = predictor.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   • Vocabulary Size: {model_info['vocabulary_size']:,}")
    print(f"   • Max Sequence Length: {model_info['max_sequence_length']}")
    print(f"   • Number of Emoji Classes: {model_info['num_classes']}")
    print(f"   • Device: {model_info['device']}")
    print(f"   • Available Emojis: {', '.join(model_info['emoji_classes'])}")
    
    # Demo sentences
    demo_sentences = [
        "I love this beautiful day",
        "This is absolutely amazing",
        "I'm feeling really sad",
        "Happy birthday to you",
        "Good night and sweet dreams",
        "I can't believe this happened",
        "This weather is perfect",
        "I need some rest now",
        "You are so awesome",
        "This pizza tastes great"
    ]
    
    print(f"\n🎯 PREDICTIONS FOR {len(demo_sentences)} SENTENCES:")
    print("-"*60)
    
    for i, sentence in enumerate(demo_sentences, 1):
        predictions = predictor.predict_emoji(sentence, top_k=3)
        
        print(f"\n{i:2d}. Sentence: '{sentence}'")
        print("    Top 3 predictions:")
        
        for j, (emoji, confidence) in enumerate(predictions, 1):
            confidence_bar = "█" * int(confidence * 20)
            print(f"       {j}. {emoji} {confidence:.1%} {confidence_bar}")
    
    print("\n" + "="*60)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\nTo use the interactive web interface:")
    print("💻 Run: streamlit run streamlit_app.py")
    print("🌐 Then open: http://localhost:8501")

def main():
    """Main demo function"""
    print("🤖 Emoji Predictor - Complete System Demo")
    print("PyTorch LSTM-RNN for Emoji Prediction")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"🚀 CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 Using CPU for inference")
    
    # Check setup
    if not check_setup():
        return
    
    # Run demo
    demo_predictions()
    
    # Interactive mode
    print(f"\n🔄 INTERACTIVE MODE")
    print("-"*30)
    print("Enter sentences to get emoji predictions (type 'quit' to exit):")
    
    try:
        predictor = EmojiPredictor()
        
        while True:
            sentence = input("\n💭 Your sentence: ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                break
            
            if sentence:
                predictions = predictor.predict_emoji(sentence, top_k=3)
                print("   🎯 Predictions:")
                for i, (emoji, conf) in enumerate(predictions, 1):
                    print(f"      {i}. {emoji} ({conf:.1%})")
            else:
                print("   ⚠️ Please enter a valid sentence")
                
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using Emoji Predictor!")
    except Exception as e:
        print(f"\n❌ Error in interactive mode: {e}")

if __name__ == "__main__":
    main()
