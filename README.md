# Emoji Predictor using LSTM-RNN 🚀

A deep learning project that predicts emojis based on text input using LSTM-RNN with PyTorch and CUDA support. Trained on social media data with comprehensive text cleaning.

## Features

- **LSTM-RNN Model**: Bidirectional LSTM for sequence processing (19.4M parameters)
- **GPU Acceleration**: CUDA support for faster training and inference
- **Interactive Web Interface**: Streamlit app with multiple input modes
- **Comprehensive Data Cleaning**: Social media text preprocessing
- **Real-time Prediction**: Live emoji prediction as you type
- **Large Vocabulary**: 12,407 words with 34 emoji classes

## Dataset

The project uses `emoji_dataset.csv` with social media text-emoji pairs. The preprocessing automatically:

- Removes URLs, mentions, and hashtags
- Cleans special characters and repeated text
- Builds comprehensive vocabulary (12,407 words)
- Encodes 34 unique emojis as labels
- Handles 37,954+ cleaned text samples

## Project Structure

```
SentenceEmoji-RNN/
├── data_preprocessing.py    # Comprehensive data cleaning and preparation
├── model_training.py        # LSTM model definition
├── gpu_training.py          # GPU-optimized training script
├── emoji_predictor.py       # Prediction module with text cleaning
├── streamlit_app.py         # Interactive web interface
├── demo.py                  # Test script with sample predictions
├── emoji_dataset.csv        # Main dataset (37,954 samples)
├── emoji_lstm_model.pth     # Trained model (19.4M parameters)
├── preprocessing_data.pkl   # Vocabulary and preprocessing data
├── requirements.txt         # Dependencies
└── old_files/              # Backup of previous versions
├── sentence_emoji_dataset.csv  # Dataset
├── emoji_lstm_model.pth     # Trained model (generated)
└── preprocessing_data.pkl   # Preprocessing objects (generated)
```

## Setup and Installation

1. **Clone the repository** (if using git)
2. **Create virtual environment**:

   ```bash
   python -m venv emoji_rnn_env
   .\emoji_rnn_env\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **For CUDA support**, install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### 1. Data Preprocessing

```bash
python data_preprocessing.py
```

### 2. Model Training (GPU Recommended)

```bash
# For GPU training (recommended)
python gpu_training.py

# For CPU training (slower)
python model_training.py
```

### 3. Test Predictions

```bash
# Run demo with sample sentences
python demo.py

# Test individual predictions
python emoji_predictor.py
```

### 4. Launch Web Interface

```bash
streamlit run streamlit_app.py
```

## Model Architecture

- **Input**: Tokenized sentences (max length: 30)
- **Embedding**: 256-dimensional word embeddings
- **LSTM**: Bidirectional, 3 layers, 512 hidden units
- **Dropout**: 0.3 for regularization
- **Output**: 34 emoji classes with confidence scores
- **Parameters**: 19.4M total parameters
- **Vocabulary**: 12,407 unique words

## Web Interface Features

### 💭 Single Sentence Mode

- Enter a sentence and get top emoji predictions
- Adjustable number of predictions (1-5)
- Confidence visualization

### 📝 Batch Prediction Mode

- Process multiple sentences at once
- Export results as CSV
- Detailed results table

### 🔄 Real-time Mode

- Live predictions as you type
- Minimum word threshold
- Instant feedback

## GPU Support

The model automatically detects and uses CUDA if available:

- **GTX 1650 Ti**: Confirmed working
- **Memory**: ~4GB VRAM sufficient
- **Fallback**: CPU training if CUDA unavailable

## Performance Notes

- **Dataset Size**: 20 unique sentences after deduplication
- **Training**: May show overfitting due to small dataset
- **Production**: Consider larger, more diverse datasets for better performance

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Streamlit for web interface
- scikit-learn for preprocessing
- pandas, numpy for data handling

## License

Open source project - feel free to modify and extend!

## Contributing

1. Add more diverse training data
2. Experiment with different architectures
3. Improve preprocessing techniques
4. Enhance the web interface
5. Add more emoji classes

---

**Happy Emoji Predicting! 😊🚀**
