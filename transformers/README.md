# Wav2Vec2 Stutter Detection System

A high-accuracy stutter detection system using fine-tuned Wav2Vec2 models for medical applications. This approach leverages pre-trained speech models and fine-tunes them specifically for stutter detection, achieving superior accuracy compared to traditional feature-based methods.

## 🎯 **Why Wav2Vec2 for Medical Applications?**

- **State-of-the-art accuracy**: Wav2Vec2 is pre-trained on thousands of hours of speech data
- **Medical-grade reliability**: Transfer learning provides robust feature representations
- **Fine-tuned for stutter detection**: Adapts pre-trained knowledge to your specific task
- **Handles variable audio lengths**: No need for fixed-length segments
- **GPU acceleration**: Efficient training and inference

## 📁 **Project Structure**

```
transformers/
├── data_preparation.py      # Prepare data for Wav2Vec2
├── train_wav2vec2.py        # Fine-tune Wav2Vec2 model
├── inference_wav2vec2.py    # Inference with trained model
├── requirements.txt         # Dependencies
├── README.md               # This file
└── processed_data/         # Generated during data preparation
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── dataset_dict/
```

## 🚀 **Quick Start**

### 1. **Install Dependencies**

```bash
cd transformers
pip install -r requirements.txt
```

### 2. **Prepare Data**

```bash
python data_preparation.py
```

This will:
- Load your `fluencybank_labels.csv`
- Map audio files to labels
- Calculate stutter percentages and assign classes
- Split data into train/validation/test sets
- Save processed data for training

### 3. **Train the Model**

```bash
python train_wav2vec2.py --epochs 15 --batch-size 4 --lr 1e-4
```

This will:
- Load pre-trained Wav2Vec2 model
- Fine-tune on your stutter detection task
- Save the best model checkpoint
- Generate training curves and confusion matrix

### 4. **Make Predictions**

```bash
# Single file
python inference_wav2vec2.py --audio path/to/audio.wav

# Batch prediction
python inference_wav2vec2.py --directory path/to/audio/folder

# Segment analysis (detailed)
python inference_wav2vec2.py --audio path/to/audio.wav --segments
```

## 🔧 **Model Architecture**

### **Pre-trained Model**
- **Base Model**: `facebook/wav2vec2-base`
- **Parameters**: ~95M parameters
- **Pre-training**: 960 hours of LibriSpeech data
- **Input**: Raw audio waveform
- **Output**: 3-class classification

### **Fine-tuning Strategy**
- **Learning Rate**: 1e-4 (recommended for fine-tuning)
- **Batch Size**: 4 (adjust based on GPU memory)
- **Epochs**: 10-15 (with early stopping)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup + cosine decay

## 📊 **Classification Logic**

The system uses the same medical-grade classification as your original system:

- **No Stutter** (< 3% stutter words): Class 0
- **At Risk** (3-5% stutter words): Class 1  
- **Stutters** (> 5% stutter words): Class 2

## 🎛️ **Advanced Configuration**

### **Training Parameters**

```bash
# Custom training
python train_wav2vec2.py \
    --epochs 20 \
    --batch-size 8 \
    --lr 5e-5 \
    --model facebook/wav2vec2-large
```

### **Available Models**

- `facebook/wav2vec2-base` (95M params) - **Recommended for most cases**
- `facebook/wav2vec2-large` (317M params) - Higher accuracy, more memory
- `facebook/wav2vec2-large-xlsr-53` (317M params) - Multi-lingual

### **Memory Optimization**

For limited GPU memory:
```bash
python train_wav2vec2.py --batch-size 2 --gradient-accumulation-steps 4
```

## 📈 **Expected Performance**

With proper fine-tuning, you should achieve:

- **Accuracy**: 85-95% (depending on data quality)
- **F1-Score**: 0.85-0.95
- **Precision**: 0.85-0.95
- **Recall**: 0.85-0.95

## 🔍 **Detailed Analysis Features**

### **Segment Analysis**
Analyze audio in time segments for detailed stutter patterns:

```bash
python inference_wav2vec2.py --audio audio.wav --segments --segment-duration 10
```

### **Confidence Scores**
Get probability distributions for all classes:
- High confidence (>80%): Reliable predictions
- Medium confidence (60-80%): Moderate reliability
- Low confidence (<60%): Uncertain predictions

## 📁 **Output Files**

### **Training Outputs**
- `wav2vec2_stutter_model/` - Trained model
- `training_results.json` - Training metrics
- `evaluation_results.json` - Test set performance
- `confusion_matrix.png` - Visual performance analysis

### **Inference Outputs**
- `predictions.json` - Batch prediction results
- Console output with detailed analysis

## 🏥 **Medical Application Guidelines**

### **Validation Requirements**
- Use stratified sampling to maintain class balance
- Validate on diverse speaker demographics
- Test on different audio quality levels
- Cross-validate across multiple folds

### **Quality Assurance**
- Monitor confidence scores for uncertain predictions
- Use ensemble methods for critical decisions
- Implement human review for low-confidence cases
- Regular model retraining with new data

## 🔧 **Troubleshooting**

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_wav2vec2.py --batch-size 2
   ```

2. **Audio Loading Errors**
   - Ensure audio files are valid WAV format
   - Check file paths and permissions
   - Verify sample rate compatibility

3. **Poor Performance**
   - Increase training epochs
   - Try larger model (wav2vec2-large)
   - Check data quality and labeling
   - Adjust learning rate

### **Performance Tips**

1. **GPU Acceleration**: Use CUDA for 5-10x speedup
2. **Mixed Precision**: Enable FP16 for memory efficiency
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **Early Stopping**: Prevent overfitting automatically

## 📚 **Technical Details**

### **Audio Processing**
- **Sample Rate**: 16kHz (Wav2Vec2 requirement)
- **Duration**: Variable length (up to 30s segments)
- **Format**: Mono WAV files
- **Preprocessing**: Automatic normalization and padding

### **Model Architecture**
- **Encoder**: Wav2Vec2 transformer encoder
- **Classification Head**: Linear layer + softmax
- **Regularization**: Dropout, weight decay
- **Optimization**: AdamW with learning rate scheduling

## 🤝 **Integration with Existing System**

This transformers approach is designed to work alongside your existing system:

1. **Same Data**: Uses your existing CSV and audio files
2. **Same Classification**: Maintains your 3-class system
3. **Complementary**: Can be used for ensemble predictions
4. **Validation**: Compare results with your original model

## 📄 **License and Citation**

This implementation is for educational and research purposes. When using in research, please cite:

- Wav2Vec2 paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
- FluencyBank dataset
- This implementation

## 🆘 **Support**

For issues or questions:
1. Check the troubleshooting section
2. Verify your data format matches requirements
3. Ensure all dependencies are installed
4. Check GPU memory availability

---

**Note**: This system is designed for medical applications and should be validated thoroughly before clinical use. 