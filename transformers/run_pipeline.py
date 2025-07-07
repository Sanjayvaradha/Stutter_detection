#!/usr/bin/env python3
"""
Complete pipeline script for Wav2Vec2 stutter detection
Runs data preparation, training, and inference in sequence
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'transformers', 'datasets', 'torch', 'librosa', 
        'pandas', 'numpy', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\nChecking data files...")
    
    required_files = [
        '../fluencybank_labels.csv',
        '../clips/clips'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - not found")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        print("Please ensure the dataset is properly set up.")
        return False
    
    return True

def run_data_preparation():
    """Run data preparation step"""
    return run_command(
        "python data_preparation.py",
        "Data Preparation"
    )

def run_training(epochs=10, batch_size=4, learning_rate=1e-4, model_name="facebook/wav2vec2-base"):
    """Run training step"""
    command = f"python train_wav2vec2.py --epochs {epochs} --batch-size {batch_size} --lr {learning_rate} --model {model_name}"
    return run_command(command, "Model Training")

def run_inference_test():
    """Run inference test on sample data"""
    # Check if model exists
    if not os.path.exists("./wav2vec2_stutter_model"):
        print("❌ Model not found. Training must be completed first.")
        return False
    
    # Check if processed data exists
    if not os.path.exists("./processed_data/test.csv"):
        print("❌ Test data not found. Data preparation must be completed first.")
        return False
    
    # Run inference on a few test files
    return run_command(
        "python inference_wav2vec2.py --directory ../clips/clips --output test_predictions.json",
        "Inference Test"
    )

def run_complete_pipeline(epochs=10, batch_size=4, learning_rate=1e-4, model_name="facebook/wav2vec2-base"):
    """Run the complete pipeline"""
    print("🚀 WAV2VEC2 STUTTER DETECTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False
    
    # Step 2: Check data files
    if not check_data_files():
        print("\n❌ Data files check failed. Please ensure dataset is properly set up.")
        return False
    
    # Step 3: Data preparation
    if not run_data_preparation():
        print("\n❌ Data preparation failed.")
        return False
    
    # Step 4: Training
    if not run_training(epochs, batch_size, learning_rate, model_name):
        print("\n❌ Training failed.")
        return False
    
    # Step 5: Inference test
    if not run_inference_test():
        print("\n❌ Inference test failed.")
        return False
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated files:")
    print("- ./processed_data/ - Processed dataset")
    print("- ./wav2vec2_stutter_model/ - Trained model")
    print("- test_predictions.json - Sample predictions")
    print("\nNext steps:")
    print("1. Use the trained model for inference:")
    print("   python inference_wav2vec2.py --audio path/to/audio.wav")
    print("2. Analyze results in test_predictions.json")
    print("3. Fine-tune parameters if needed")
    
    return True

def run_quick_test():
    """Run a quick test with minimal training"""
    print("🧪 QUICK TEST MODE - Minimal training for testing")
    return run_complete_pipeline(epochs=2, batch_size=2, learning_rate=1e-4)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Wav2Vec2 Stutter Detection Pipeline')
    parser.add_argument('--mode', choices=['complete', 'quick', 'data-only', 'train-only'], 
                       default='complete', help='Pipeline mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base', 
                       help='Pre-trained model name')
    
    args = parser.parse_args()
    
    if args.mode == 'complete':
        run_complete_pipeline(args.epochs, args.batch_size, args.lr, args.model)
    
    elif args.mode == 'quick':
        run_quick_test()
    
    elif args.mode == 'data-only':
        if check_dependencies() and check_data_files():
            run_data_preparation()
    
    elif args.mode == 'train-only':
        if check_dependencies():
            run_training(args.epochs, args.batch_size, args.lr, args.model)

if __name__ == "__main__":
    main() 