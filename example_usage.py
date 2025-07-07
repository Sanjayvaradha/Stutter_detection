#!/usr/bin/env python3
"""
Example usage of the Stutter Detection System
"""

import os
from stutter_detection import StutterDetector

def example_training():
    """Example of training the model"""
    print("=== Example: Training the Model ===")
    
    # Initialize the detector
    detector = StutterDetector()
    
    # Run the complete training pipeline
    detector.run_complete_pipeline()
    
    print("Training completed! Check the generated files:")
    print("- best_stutter_model.pth (trained model)")
    print("- training_curves.png (training visualization)")
    print("- confusion_matrix.png (performance visualization)")

def example_single_prediction():
    """Example of single file prediction"""
    print("\n=== Example: Single File Prediction ===")
    
    # Check if model exists
    if not os.path.exists('best_stutter_model.pth'):
        print("❌ Model not found. Please train the model first.")
        return
    
    # Initialize detector
    detector = StutterDetector()
    
    # Example: predict for a sample audio file
    sample_audio = "clips/clips/FluencyBank_010_0.wav"
    
    if os.path.exists(sample_audio):
        print(f"Predicting for: {sample_audio}")
        result = detector.predict_single_audio(sample_audio)
        
        if result:
            print(f"Result: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.2%}")
    else:
        print(f"❌ Sample audio file not found: {sample_audio}")

def example_batch_prediction():
    """Example of batch prediction"""
    print("\n=== Example: Batch Prediction ===")
    
    # Check if model exists
    if not os.path.exists('best_stutter_model.pth'):
        print("❌ Model not found. Please train the model first.")
        return
    
    # Initialize detector
    detector = StutterDetector()
    
    # Example: predict for first 5 audio files
    audio_dir = "clips/clips"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')][:5]
        
        print(f"Predicting for {len(audio_files)} sample files...")
        
        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            result = detector.predict_single_audio(audio_path)
            
            if result:
                print(f"{audio_file}: {result['class_name']} ({result['confidence']:.1%})")
    else:
        print(f"❌ Audio directory not found: {audio_dir}")

def example_custom_thresholds():
    """Example of using custom classification thresholds"""
    print("\n=== Example: Custom Thresholds ===")
    
    # You can modify the thresholds in the StutterDetector class
    print("To use custom thresholds, modify the load_and_preprocess_data method:")
    print("""
    # Current thresholds:
    if stutter_percentage < 3:
        final_label = 0  # No stutter
    elif stutter_percentage <= 5:
        final_label = 1  # At risk
    else:
        final_label = 2  # Stutters
    
    # Custom thresholds example:
    if stutter_percentage < 2:
        final_label = 0  # No stutter
    elif stutter_percentage <= 4:
        final_label = 1  # At risk
    else:
        final_label = 2  # Stutters
    """)

def main():
    """Main example function"""
    print("Stutter Detection System - Example Usage\n")
    
    # Check if we're in the right directory
    if not os.path.exists('stutter_detection.py'):
        print("❌ Please run this script from the project directory")
        return
    
    # Show available examples
    print("Available examples:")
    print("1. Training the model")
    print("2. Single file prediction")
    print("3. Batch prediction")
    print("4. Custom thresholds")
    
    choice = input("\nEnter your choice (1-4) or 'all' to run all examples: ").strip()
    
    if choice == '1' or choice == 'all':
        example_training()
    
    if choice == '2' or choice == 'all':
        example_single_prediction()
    
    if choice == '3' or choice == 'all':
        example_batch_prediction()
    
    if choice == '4' or choice == 'all':
        example_custom_thresholds()
    
    print("\n=== Example Usage Complete ===")

if __name__ == "__main__":
    main() 