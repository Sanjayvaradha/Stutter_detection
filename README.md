# Stutter Detection System

A comprehensive machine learning system for detecting stuttering in audio files. The system classifies speech into three categories:
- **No Stutter** (< 3% stutter words)
- **At Risk** (3-5% stutter words)  
- **Stutters** (> 5% stutter words)

## Features

- **Audio Feature Extraction**: MFCC, spectral features, zero crossing rate, RMS energy, chroma features
- **Deep Learning Model**: Multi-layer neural network with dropout for regularization
- **Data Preprocessing**: Automatic audio-label mapping, data cleaning, train/validation/test split
- **Training Pipeline**: Complete training with validation, early stopping, and model saving
- **Inference**: Single file and batch prediction capabilities
- **Visualization**: Training curves and confusion matrix plots

## Dataset Structure

The system expects:
- Audio files: `FluencyBank_{EpId}_{ClipId}.wav` in `clips/clips/` directory
- Labels file: `fluencybank_labels.csv` with columns for different stutter types

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify dataset structure**:
```
stutter_dataset/
├── clips/
│   └── clips/
│       ├── FluencyBank_010_0.wav
│       ├── FluencyBank_010_1.wav
│       └── ...
└── fluencybank_labels.csv
```

## Usage

### 1. Training the Model

Run the complete training pipeline:

```bash
python stutter_detection.py
```

This will:
- Load and preprocess the dataset
- Split data into train/validation/test sets
- Train the neural network model
- Save the best model as `best_stutter_model.pth`
- Generate training curves and confusion matrix plots

### 2. Single File Prediction

Predict stutter classification for a single audio file:

```bash
python inference.py --audio path/to/audio.wav
```

### 3. Batch Prediction

Predict stutter classification for all audio files in a directory:

```bash
python inference.py --directory path/to/audio/folder
```

### 4. Using Custom Model

Specify a different trained model:

```bash
python inference.py --audio path/to/audio.wav --model path/to/model.pth
```

## Model Architecture

The neural network consists of:
- Input layer: 128 features (audio features)
- Hidden layers: 256 → 128 → 64 neurons with ReLU activation
- Dropout layers: 30% dropout for regularization
- Output layer: 3 classes with softmax activation

## Audio Features Extracted

1. **MFCC Features**: 13 MFCC coefficients (mean and std)
2. **Spectral Features**: Centroid, rolloff, bandwidth (mean and std)
3. **Zero Crossing Rate**: Mean and standard deviation
4. **RMS Energy**: Mean and standard deviation
5. **Chroma Features**: 12 chroma coefficients (mean)

## Classification Logic

The system calculates stutter percentage based on:
- Stutter types: Prolongation, Block, SoundRep, WordRep
- Total words: NoStutteredWords column
- Formula: `(total_stutters / total_words) * 100`

Classification thresholds:
- **No Stutter**: < 3%
- **At Risk**: 3-5%
- **Stutters**: > 5%

## Output Files

After training, the system generates:
- `best_stutter_model.pth`: Trained model weights
- `training_curves.png`: Training and validation loss plots
- `confusion_matrix.png`: Confusion matrix visualization

## Performance Metrics

The system provides:
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Training/validation loss curves
- Prediction confidence scores

## Customization

### Modify Feature Extraction

Edit the `extract_features` method in `StutterDataset` class to add/remove features.

### Adjust Model Architecture

Modify the `StutterClassifier` class to change:
- Number of layers
- Hidden layer sizes
- Activation functions
- Dropout rates

### Change Classification Thresholds

Update the percentage thresholds in the `load_and_preprocess_data` method.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `create_data_loaders`
2. **Audio loading errors**: Check audio file format and path
3. **Model not found**: Ensure training completed successfully

### Performance Tips

1. **GPU Acceleration**: The system automatically uses CUDA if available
2. **Batch Processing**: Use batch prediction for multiple files
3. **Memory Management**: Adjust batch size based on available memory

## Dependencies

- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **TQDM**: Progress bars

## License

This project is for educational and research purposes.

## Citation

If you use this system in your research, please cite the FluencyBank dataset and this implementation. 