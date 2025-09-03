# Emoji Predictor using LSTM-RNN 🚀

A deep learning project that predicts emojis based on text input using LSTM-RNN with PyTorch and CUDA support.

## Features

- **LSTM-RNN Model**: Bidirectional LSTM for sequence processing
- **GPU Acceleration**: CUDA support for faster training and inference
- **Interactive Web Interface**: Streamlit app with multiple input modes
- **Data Preprocessing**: Automated cleaning and vocabulary building
- **Real-time Prediction**: Live emoji prediction as you type

## Dataset

The project uses `sentence_emoji_dataset.csv` with sentence-emoji pairs. The preprocessing automatically:

- Removes duplicate sentences
- Handles missing values
- Builds vocabulary from text
- Encodes emojis as labels

## Project Structure

```
SentenceEmoji-RNN/
├── data_preprocessing.py    # Data cleaning and preparation
├── model_training.py        # LSTM model training
├── emoji_predictor.py       # Prediction module
├── streamlit_app.py         # Web interface
├── requirements.txt         # Dependencies
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

### 2. Train the Model

```bash
python model_training.py
```

### 3. Test Predictions

```bash
python emoji_predictor.py
```

### 4. Launch Web Interface

```bash
streamlit run streamlit_app.py
```

## Model Architecture

- **Input**: Tokenized sentences (max length: 20)
- **Embedding**: 128-dimensional word embeddings
- **LSTM**: Bidirectional, 2 layers, 256 hidden units
- **Output**: 13 emoji classes with confidence scores

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
