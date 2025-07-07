#!/usr/bin/env python3
"""
Inference script for fine-tuned Wav2Vec2 stutter detection model
"""

import os
import torch
import numpy as np
import librosa
import json
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

class Wav2Vec2StutterInference:
    """Inference class for Wav2Vec2 stutter detection"""
    
    def __init__(self, model_path="./wav2vec2_stutter_model", sample_rate=16000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.load_model()
        
        # Class names
        self.class_names = ['No Stutter', 'At Risk', 'Stutters']
        
    def load_model(self):
        """Load the fine-tuned model and processor"""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Load processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model has been trained and saved correctly.")
            raise
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file for inference"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Pad or truncate to 30 seconds
            max_length = 30 * self.sample_rate
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            
            # Process with Wav2Vec2 processor
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            return inputs.input_values.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing {audio_path}: {e}")
            return None
    
    def predict_single(self, audio_path):
        """Predict stutter classification for a single audio file"""
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None
        
        # Preprocess audio
        inputs = self.preprocess_audio(audio_path)
        if inputs is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare result
        result = {
            'file': os.path.basename(audio_path),
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
        
        return result
    
    def predict_batch(self, audio_directory):
        """Predict stutter classification for all audio files in a directory"""
        if not os.path.exists(audio_directory):
            print(f"Audio directory not found: {audio_directory}")
            return []
        
        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.m4b']
        audio_files = []
        
        for file in os.listdir(audio_directory):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(audio_directory, file))
        
        if not audio_files:
            print("No audio files found in the directory.")
            return []
        
        print(f"Found {len(audio_files)} audio files for prediction.")
        
        # Process all files
        results = []
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            result = self.predict_single(audio_file)
            if result:
                results.append(result)
        
        return results
    
    def print_prediction_result(self, result):
        """Print prediction result in a formatted way"""
        if result is None:
            return
        
        print(f"\n{'='*50}")
        print(f"File: {result['file']}")
        print(f"Prediction: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        for i, (class_name, prob) in enumerate(zip(self.class_names, result['probabilities'])):
            print(f"  {class_name}: {prob:.2%}")
        print(f"{'='*50}")
    
    def print_batch_summary(self, results):
        """Print summary of batch prediction results"""
        if not results:
            return
        
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")
        
        # Count predictions
        class_counts = {class_name: 0 for class_name in self.class_names}
        total_confidence = 0
        
        for result in results:
            class_counts[result['class_name']] += 1
            total_confidence += result['confidence']
        
        # Print statistics
        print(f"Total files processed: {len(results)}")
        print(f"Average confidence: {total_confidence/len(results):.2%}")
        print(f"\nClass Distribution:")
        
        for class_name, count in class_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {class_name}: {count} files ({percentage:.1f}%)")
        
        # Show high-confidence predictions
        high_conf_threshold = 0.8
        high_conf_results = [r for r in results if r['confidence'] > high_conf_threshold]
        
        if high_conf_results:
            print(f"\nHigh-confidence predictions (>80%):")
            for result in high_conf_results[:10]:  # Show first 10
                print(f"  {result['file']}: {result['class_name']} ({result['confidence']:.1%})")
            
            if len(high_conf_results) > 10:
                print(f"  ... and {len(high_conf_results) - 10} more")
        
        print(f"{'='*60}")
    
    def save_results(self, results, output_file):
        """Save prediction results to JSON file"""
        if not results:
            return
        
        # Add metadata
        output_data = {
            'metadata': {
                'model_path': self.model_path,
                'total_files': len(results),
                'device': str(self.device),
                'class_names': self.class_names
            },
            'predictions': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    def analyze_audio_segments(self, audio_path, segment_duration=10):
        """Analyze audio in segments for detailed stutter analysis"""
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Load full audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Calculate segment length in samples
            segment_length = segment_duration * self.sample_rate
            
            # Split audio into segments
            segments = []
            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                if len(segment) > 0:
                    segments.append(segment)
            
            print(f"Analyzing {len(segments)} segments of {segment_duration}s each...")
            
            # Analyze each segment
            segment_results = []
            for i, segment in enumerate(segments):
                # Pad segment if needed
                if len(segment) < segment_length:
                    segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
                
                # Process segment
                inputs = self.processor(
                    segment, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).input_values.to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                segment_results.append({
                    'segment': i + 1,
                    'start_time': i * segment_duration,
                    'end_time': (i + 1) * segment_duration,
                    'predicted_class': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy().tolist()
                })
            
            return segment_results
            
        except Exception as e:
            print(f"Error analyzing segments: {e}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Wav2Vec2 Stutter Detection Inference')
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--directory', type=str, help='Path to directory with audio files')
    parser.add_argument('--model', type=str, default='./wav2vec2_stutter_model', 
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for batch predictions')
    parser.add_argument('--segments', action='store_true',
                       help='Analyze audio in segments (for single file only)')
    parser.add_argument('--segment-duration', type=int, default=10,
                       help='Duration of segments in seconds')
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        inference = Wav2Vec2StutterInference(model_path=args.model)
    except Exception as e:
        print(f"Failed to initialize inference: {e}")
        return
    
    if args.audio:
        # Single file prediction
        result = inference.predict_single(args.audio)
        inference.print_prediction_result(result)
        
        if args.segments and result:
            print(f"\nAnalyzing audio segments...")
            segment_results = inference.analyze_audio_segments(
                args.audio, args.segment_duration
            )
            
            if segment_results:
                print(f"\nSegment Analysis Results:")
                for segment in segment_results:
                    print(f"Segment {segment['segment']} ({segment['start_time']}-{segment['end_time']}s): "
                          f"{segment['class_name']} ({segment['confidence']:.1%})")
    
    elif args.directory:
        # Batch prediction
        results = inference.predict_batch(args.directory)
        inference.print_batch_summary(results)
        
        # Save results
        if results:
            inference.save_results(results, args.output)
    
    else:
        print("Please provide either --audio for single file or --directory for batch prediction")
        print("Example usage:")
        print("  python inference_wav2vec2.py --audio path/to/audio.wav")
        print("  python inference_wav2vec2.py --directory path/to/audio/folder")
        print("  python inference_wav2vec2.py --audio path/to/audio.wav --segments")

if __name__ == "__main__":
    main() 