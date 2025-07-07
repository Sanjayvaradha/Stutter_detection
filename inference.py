import os
import torch
import numpy as np
from stutter_detection import StutterDetector, StutterDataset

def predict_stutter(audio_path, model_path='best_stutter_model.pth'):
    """
    Predict stutter classification for a single audio file
    
    Args:
        audio_path (str): Path to the audio file
        model_path (str): Path to the trained model
    
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return None
    
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found.")
        return None
    
    # Initialize detector and load model
    detector = StutterDetector()
    detector.model = detector.model = detector.model.__class__().to(detector.device)
    detector.model.load_state_dict(torch.load(model_path, map_location=detector.device))
    
    # Make prediction
    result = detector.predict_single_audio(audio_path)
    
    if result:
        print(f"\nPrediction Results for {os.path.basename(audio_path)}:")
        print(f"Classification: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        for i, (class_name, prob) in enumerate(zip(class_names, result['probabilities'])):
            print(f"  {class_name}: {prob:.2%}")
    
    return result

def batch_predict(audio_directory, model_path='best_stutter_model.pth'):
    """
    Predict stutter classification for all audio files in a directory
    
    Args:
        audio_directory (str): Path to directory containing audio files
        model_path (str): Path to the trained model
    
    Returns:
        list: List of prediction results
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return []
    
    if not os.path.exists(audio_directory):
        print(f"Audio directory {audio_directory} not found.")
        return []
    
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    for file in os.listdir(audio_directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(audio_directory, file))
    
    if not audio_files:
        print("No audio files found in the directory.")
        return []
    
    print(f"Found {len(audio_files)} audio files for prediction.")
    
    # Initialize detector and load model
    detector = StutterDetector()
    detector.model = detector.model.__class__().to(detector.device)
    detector.model.load_state_dict(torch.load(model_path, map_location=detector.device))
    
    results = []
    
    for audio_file in audio_files:
        print(f"\nProcessing: {os.path.basename(audio_file)}")
        result = detector.predict_single_audio(audio_file)
        if result:
            result['file'] = os.path.basename(audio_file)
            results.append(result)
    
    # Print summary
    print(f"\n=== Batch Prediction Summary ===")
    class_counts = {'No Stutter': 0, 'At Risk': 0, 'Stutters': 0}
    
    for result in results:
        class_counts[result['class_name']] += 1
    
    for class_name, count in class_counts.items():
        percentage = (count / len(results)) * 100 if results else 0
        print(f"{class_name}: {count} files ({percentage:.1f}%)")
    
    return results

def main():
    """Main function for inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stutter Detection Inference')
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--directory', type=str, help='Path to directory with audio files')
    parser.add_argument('--model', type=str, default='best_stutter_model.pth', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.audio:
        predict_stutter(args.audio, args.model)
    elif args.directory:
        batch_predict(args.directory, args.model)
    else:
        print("Please provide either --audio for single file or --directory for batch prediction")
        print("Example usage:")
        print("  python inference.py --audio path/to/audio.wav")
        print("  python inference.py --directory path/to/audio/folder")

if __name__ == "__main__":
    main() 