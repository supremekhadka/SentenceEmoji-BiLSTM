# Emoji Predictor using LSTM-RNN

![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green)

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
├── Core Model Files
│   ├── emoji_lstm_model.pth      # Trained model (19.4M parameters)
│   ├── preprocessing_data.pkl    # Vocabulary and preprocessing data
│   └── emoji_dataset.csv         # Main dataset (37,954 samples)
├── Processing & Training
│   ├── data_preprocessing.py     # Comprehensive data cleaning and preparation
│   ├── model_training.py         # LSTM model definition
│   └── gpu_training.py           # GPU-optimized training script
├── Prediction & Interface
│   ├── emoji_predictor.py        # Prediction module with text cleaning
│   ├── streamlit_app.py          # Interactive web interface
│   └── demo.py                   # Test script with sample predictions
├── Documentation & Config
│   ├── README.md                 # Project documentation
│   ├── project_summary.py        # Project analysis and system info
│   ├── requirements.txt          # Dependencies
│   └── .gitignore               # Git ignore rules
└── Environment
    └── emoji_rnn_env/            # Virtual environment
```

## Setup and Installation

1. **Clone the repository** (if using git)
2. **Create virtual environment**:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **For CUDA support**, install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio 
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

### Single Sentence Mode

- Enter a sentence and get top emoji predictions
- Adjustable number of predictions (1-5)
- Confidence visualization

### Batch Prediction Mode

- Process multiple sentences at once
- Export results as CSV
- Detailed results table

### Real-time Mode

- Live predictions as you type
- Minimum word threshold for meaningful predictions
- Instant feedback and confidence visualization

## Example Predictions

The model excels at understanding emotional context and social media language:

```
Input: "I love this so much it makes me so happy"
Output: 🥹 (98.1%) 🥲 (1.4%) 😭 (0.3%)

Input: "Party time let's celebrate tonight"
Output: 🎉 (100%) ✅ (0.0%) 🥳 (0.0%)

Input: "Happy birthday hope you have an amazing day"
Output: 🥳 (82.4%) 🎉 (4.3%) 🐣 (3.1%)

Input: "Missing my family so much right now"
Output: 😭 (43.9%) 💀 (10.0%) 🤣 (9.6%)
```

## Available Emoji Classes

The model predicts from 34 emoji classes commonly used in social media:

- **Emotions**: 😊 😭 🥹 🥲 🤣 😤 😡 😎 😉
- **Celebrations**: 🎉 🥳 ✨ 🙏
- **Activities**: ✅ 🔥 👀 👻
- **Expressions**: 🫠 🤍 🥰 🐰 🐣
- **And more**: Covering diverse social media emotions and reactions

## GPU Support

The model automatically detects and uses CUDA if available:

- **GTX 1650 Ti**: Confirmed working
- **Memory**: ~4GB VRAM sufficient
- **Fallback**: CPU training if CUDA unavailable

## Performance Notes

- **Dataset Size**: 37,954 cleaned social media text samples
- **Vocabulary**: 12,407 unique words with comprehensive cleaning
- **Model Size**: 19.4M parameters for robust emoji prediction
- **Training Time**: ~40 minutes on GTX 1650 Ti GPU
- **Accuracy**: High confidence predictions (95%+ for clear emotional expressions)
- **Production Ready**: Large-scale dataset with comprehensive preprocessing

## Recent Updates

### Project Reorganization (September 2025)

- **Cleaned Architecture**: Removed "twitter" references for generic social media use
- **Unified Codebase**: Single clean prediction pipeline
- **Enhanced Documentation**: Comprehensive README and project summary
- **Streamlined Files**: Professional file naming and organization

### Model Improvements

- **Scaled Architecture**: Increased to 3-layer bidirectional LSTM (512 hidden units)
- **Expanded Dataset**: From 20 samples to 37,954+ social media texts
- **Enhanced Cleaning**: Comprehensive text preprocessing pipeline
- **GPU Optimization**: Full CUDA support with memory management

## System Requirements

- **Python**: 3.8+ (tested with 3.12)
- **PyTorch**: 2.0+ with CUDA 11.8+ for GPU acceleration
- **Memory**: 4GB+ RAM, 2GB+ GPU memory (for training)
- **Storage**: 1GB+ for model and dataset files
- **OS**: Windows/Linux/MacOS (PowerShell scripts included for Windows)

## Dependencies

Core packages (see `requirements.txt` for complete list):

- **PyTorch 2.7.1+cu118**: Deep learning framework with CUDA support
- **Streamlit**: Interactive web interface
- **scikit-learn**: Preprocessing and encoding
- **pandas & numpy**: Data manipulation
- **matplotlib**: Visualization for training metrics

## Quick Start

1. **Clone and setup**:

   ```bash
   git clone <repository-url>
   cd SentenceEmoji-RNN
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Test the model**:

   ```bash
   python demo.py                    # Run sample predictions
   python project_summary.py         # View system info
   ```

3. **Launch web interface**:
   ```bash
   streamlit run streamlit_app.py
   # Open http://localhost:8501
   ```

## Troubleshooting

### Common Issues

- **CUDA not found**: Install PyTorch with CUDA or use CPU training
- **Memory errors**: Reduce batch size in `gpu_training.py`
- **Emoji display issues**: Console encoding; use Streamlit interface instead
- **Model not found**: Ensure `emoji_lstm_model.pth` exists in project root

### Performance Tips

- **GPU Training**: Use `gpu_training.py` for 10x faster training
- **Batch Size**: Increase for GPU, decrease for CPU/low memory
- **Real-time Mode**: Works best with 3+ word sentences

## License

Open source project

## Contributing

The project is production-ready but welcomes improvements:

1. **Data Enhancement**: Add multilingual social media datasets
2. **Model Architecture**: Experiment with Transformers or attention mechanisms
3. **Preprocessing**: Improve text cleaning for other social platforms
4. **Interface**: Enhance Streamlit UI with themes and animations
5. **Performance**: Optimize for mobile deployment or edge devices
6. **Features**: Add emoji sentiment analysis or emoji-to-text generation

## Project Status

**Complete and Production-Ready**

- Trained model with 19.4M parameters
- Comprehensive dataset (37,954 samples)
- Full web interface with three interaction modes
- Professional codebase with clean architecture
- Detailed documentation and troubleshooting guides

## Acknowledgments

- Built with PyTorch and Streamlit
- Trained on social media emoji usage patterns
- Optimized for GPU acceleration (GTX 1650 Ti tested)
- Comprehensive text preprocessing pipeline

---
