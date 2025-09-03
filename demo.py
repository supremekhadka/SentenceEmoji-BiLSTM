"""
Demo script to test the emoji predictor with sample sentences
"""

from emoji_predictor import EmojiPredictor
import json

def test_samples():
    """Test the predictor with various sample sentences"""
    
    # Initialize predictor
    predictor = EmojiPredictor()
    
    # Sample sentences that are typical of social media content
    test_samples = [
        # Positive emotions
        "I love this so much it makes me so happy",
        "This is absolutely amazing and incredible",
        "Feeling blessed and grateful today",
        "Best day ever can't stop smiling",
        
        # Celebrations
        "Happy birthday hope you have an amazing day",
        "Congratulations on your achievement",
        "Party time let's celebrate tonight",
        "Just got promoted at work so excited",
        
        # Food and lifestyle
        "This pizza looks absolutely delicious",
        "Coffee and good vibes this morning",
        "Workout done feeling strong and energized",
        "Beautiful sunset perfect for photos",
        
        # Sad/emotional
        "Having a tough day feeling really down",
        "Missing my family so much right now",
        "Can't believe this happened to me",
        
        # Weather and nature
        "Perfect weather for a beach day",
        "Love rainy days perfect for staying cozy",
        "Spring flowers are blooming everywhere beautiful",
        
        # Social
        "Good night sweet dreams everyone",
        "Thank you for all the support and love",
        "Friends make everything better grateful"
    ]
    
    print("🚀 Testing Emoji Predictor")
    print("="*70)
    print(f"Model loaded with {predictor.get_model_info()['vocabulary_size']:,} vocabulary words")
    print(f"Predicting for {len(test_samples)} sample sentences...")
    print("="*70)
    
    results = []
    
    for i, sentence in enumerate(test_samples, 1):
        # Get predictions
        predictions = predictor.predict_emoji(sentence, top_k=3)
        cleaned = predictor.clean_text(sentence)
        
        # Store results
        result = {
            'original': sentence,
            'cleaned': cleaned,
            'predictions': [(emoji, f"{conf:.3f}") for emoji, conf in predictions]
        }
        results.append(result)
        
        # Display
        print(f"{i:2d}. '{sentence}'")
        print(f"    Cleaned: '{cleaned}'")
        print("    Predictions:", end=" ")
        for j, (emoji, confidence) in enumerate(predictions):
            print(f"{emoji} ({confidence:.3f})", end="")
            if j < len(predictions) - 1:
                print(", ", end="")
        print("\n")
    
    # Save results for analysis
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Test completed! Results saved to 'test_results.json'")
    
    # Summary stats
    all_top_predictions = [result['predictions'][0][0] for result in results]
    unique_emojis = set(all_top_predictions)
    
    print(f"\n📊 Summary:")
    print(f"- Total sentences tested: {len(test_samples)}")
    print(f"- Unique top predictions: {len(unique_emojis)}")
    print(f"- Most common predictions: {', '.join(list(unique_emojis)[:10])}")

if __name__ == "__main__":
    test_samples()
