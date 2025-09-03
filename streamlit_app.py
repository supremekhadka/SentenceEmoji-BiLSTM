import streamlit as st
import torch
import pandas as pd
import numpy as np
from emoji_predictor import EmojiPredictor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="VibeCheck",
    page_icon="🐈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emoji-result {
        font-size: 2rem;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the Twitter emoji predictor model (cached)"""
    try:
        predictor = EmojiPredictor()
        return predictor, True
    except Exception as e:
        return str(e), False

def display_predictions(predictions, sentence):
    """Display prediction results in a nice format"""
    if not predictions:
        st.error("No predictions available")
        return
    
    st.markdown("### 🎯 Prediction Results")
    
    # Create columns for the top 3 predictions
    cols = st.columns(3)
    
    for i, (emoji, confidence) in enumerate(predictions[:3]):
        with cols[i]:
            # Emoji display
            st.markdown(f"""
            <div class="emoji-result" style="background-color: {'#e8f5e8' if i==0 else '#f0f8ff' if i==1 else '#fff5ee'}">
                <div style="font-size: 3rem;">{emoji}</div>
                <div style="font-size: 1rem; color: #666;">#{i+1} Choice</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(confidence)

def create_confidence_chart(predictions):
    """Create a confidence chart for predictions"""
    if not predictions:
        return None
    
    emojis = [pred[0] for pred in predictions]
    confidences = [pred[1] * 100 for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=emojis,
            orientation='h',
            marker=dict(
                color=confidences,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence %")
            ),
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Emoji Prediction Confidence",
        xaxis_title="Confidence (%)",
        yaxis_title="Emoji",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">VibeCheck</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict emojis from your sentences using Bi-LSTM</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading emoji prediction model..."):
        predictor, success = load_predictor()
    
    if not success:
        st.error(f"❌ Failed to load model: {predictor}")
        st.info("Please make sure you have trained the model first by running `python model_training.py`")
        return
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        
        model_info = predictor.get_model_info()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Model Stats</h4>
            <p><strong>Vocabulary Size:</strong> {model_info['vocabulary_size']:,}</p>
            <p><strong>Max Sequence Length:</strong> {model_info['max_sequence_length']}</p>
            <p><strong>Number of Emoji Classes:</strong> {model_info['num_classes']}</p>
            <p><strong>Device:</strong> {model_info['device']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 😊 Available Emojis")
        emoji_cols = st.columns(2)
        for i, emoji in enumerate(model_info['emoji_classes']):
            with emoji_cols[i % 2]:
                st.markdown(f"**{emoji}**")
    
    # Main interface
    st.markdown("---")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["💭 Type a sentence", "📝 Batch prediction", "🔄 Real-time typing"],
        horizontal=True
    )
    
    if input_method == "💭 Type a sentence":
        single_sentence_mode(predictor)
    elif input_method == "📝 Batch prediction":
        batch_prediction_mode(predictor)
    else:
        real_time_mode(predictor)

def single_sentence_mode(predictor):
    """Single sentence prediction mode"""
    st.markdown("### 💭 Single Sentence Prediction")
    
    # Number of predictions
    top_k = st.slider("Number of predictions:", 1, 5, 3)
    
    # Text input with form for Enter key support
    with st.form(key="prediction_form", clear_on_submit=False):
        sentence = st.text_input(
            "Enter your sentence and press Enter to predict:",
            placeholder="e.g., I love this beautiful day!",
            help="Type any sentence and press Enter to get emoji predictions"
        )
        
        # Form submit button (triggered by Enter key)
        predict_button = st.form_submit_button("🎯 Predict Emoji", type="primary")
    
    # Predict when form is submitted (Enter pressed or button clicked)
    if predict_button and sentence:
        with st.spinner("Predicting..."):
            predictions = predictor.predict_emoji(sentence, top_k)
            
            # Display results
            display_predictions(predictions, sentence)
            
            # Confidence chart
            if len(predictions) > 1:
                st.markdown("### 📊 Confidence Chart")
                fig = create_confidence_chart(predictions)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            results_df = pd.DataFrame([
                {"Rank": i+1, "Emoji": emoji, "Confidence": f"{conf:.1%}"}
                for i, (emoji, conf) in enumerate(predictions)
            ])
            st.dataframe(results_df, use_container_width=True)
    elif predict_button and not sentence:
        st.warning("⚠️ Please enter a sentence first!")

def batch_prediction_mode(predictor):
    """Batch prediction mode"""
    st.markdown("### 📝 Batch Prediction")
    
    top_k = st.slider("Number of predictions per sentence:", 1, 5, 3, key="batch_top_k")
    
    # Use form for batch prediction too
    with st.form(key="batch_form", clear_on_submit=False):
        sentences_text = st.text_area(
            "Enter sentences (one per line) and press Enter to predict:",
            placeholder="I love this day!\nThis is amazing!\nI'm feeling sad today.",
            height=150,
            help="Enter multiple sentences, one per line, then press Enter to predict all"
        )
        
        # Form submit button
        predict_button = st.form_submit_button("🎯 Predict All", type="primary")
    
    if predict_button and sentences_text:
        sentences = [s.strip() for s in sentences_text.split('\n') if s.strip()]
        
        if sentences:
            with st.spinner(f"Predicting for {len(sentences)} sentences..."):
                batch_predictions = predictor.predict_batch(sentences, top_k)
                
                st.markdown("### 📊 Batch Results")
                
                # Create results table
                results_data = []
                for i, (sentence, predictions) in enumerate(zip(sentences, batch_predictions)):
                    for j, (emoji, conf) in enumerate(predictions):
                        results_data.append({
                            "Sentence": sentence,
                            "Rank": j + 1,
                            "Emoji": emoji,
                            "Confidence": conf
                        })
                
                results_df = pd.DataFrame(results_data)
                
                # Display results by sentence
                for i, sentence in enumerate(sentences):
                    with st.expander(f"📝 Sentence {i+1}: '{sentence}'"):
                        sentence_preds = batch_predictions[i]
                        display_predictions(sentence_preds, sentence)
                
                # Overall results table
                st.markdown("### 📋 Complete Results Table")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"emoji_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Please enter at least one sentence!")
    elif predict_button and not sentences_text:
        st.warning("⚠️ Please enter some sentences first!")

def real_time_mode(predictor):
    """Real-time typing mode"""
    st.markdown("### 🔄 Real-time Prediction")
    st.info("🔥 Type and see predictions update automatically in real-time!")
    
    # Real-time prediction threshold
    min_words = st.slider("Minimum words for prediction:", 1, 5, 2)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input that triggers updates on every keystroke
        sentence = st.text_input(
            "Start typing:",
            placeholder="Start typing your sentence...",
            key="rt_input_main",
            help="Predictions will update automatically as you type"
        )
    
    with col2:
        # Show real-time stats
        if sentence:
            st.metric("Words", len(sentence.split()))
            st.metric("Characters", len(sentence))
    
    # Real-time predictions
    if sentence and len(sentence.split()) >= min_words:
        try:
            # Show loading indicator briefly
            with st.spinner("🔄 Updating..."):
                predictions = predictor.predict_emoji(sentence, 3)
            
            # Create two columns for predictions
            pred_col1, pred_col2 = st.columns([2, 1])
            
            with pred_col1:
                st.markdown("#### 🎯 Live Predictions:")
                
                # Display predictions in a grid
                for i, (emoji, conf) in enumerate(predictions):
                    col_emoji, col_conf = st.columns([1, 3])
                    with col_emoji:
                        st.markdown(f"### {emoji}")
                    with col_conf:
                        st.metric(f"Rank #{i+1}", f"{conf:.1%}")
                        st.progress(conf)
            
            with pred_col2:
                # Show top prediction prominently
                top_emoji, top_conf = predictions[0]
                st.markdown("#### 🏆 Top Pick:")
                st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{top_emoji}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 1.5rem;'>{top_conf:.1%}</div>", 
                           unsafe_allow_html=True)
            
            # Additional analysis
            with st.expander("📈 Real-time Analysis", expanded=False):
                st.write(f"**Current sentence:** '{sentence}'")
                st.write(f"**Tokenized words:** {sentence.split()}")
                
                # Show all predictions in a table
                pred_df = pd.DataFrame([
                    {"Rank": i+1, "Emoji": emoji, "Confidence": f"{conf:.2%}"}
                    for i, (emoji, conf) in enumerate(predictions)
                ])
                st.dataframe(pred_df, hide_index=True)
                
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            
    elif sentence and len(sentence.split()) < min_words:
        st.info(f"💭 Type at least {min_words} words to see live predictions...")
        st.markdown(f"**Current words:** {len(sentence.split())}/{min_words}")
        
    elif not sentence:
        st.info("💭 Start typing to see real-time emoji predictions!")
        
        # Show some example sentences
        st.markdown("#### 💡 Try these examples:")
        examples = [
            "I love this beautiful day",
            "This is absolutely amazing", 
            "I'm feeling really sad",
            "Happy birthday to you"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    # Use a different approach to set the text
                    st.info(f"Try typing: '{example}'")
    
    # Auto-refresh for better real-time experience
    if sentence:
        time.sleep(0.1)  # Small delay to prevent too frequent updates

if __name__ == "__main__":
    main()
