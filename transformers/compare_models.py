#!/usr/bin/env python3
"""
Model comparison script
Compares the performance of the original feature-based model vs Wav2Vec2 model
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import original model
sys.path.append('..')
try:
    from stutter_detection import StutterDetector
except ImportError:
    print("Warning: Could not import original model. Make sure stutter_detection.py exists in parent directory.")

class ModelComparator:
    """Compare different stutter detection models"""
    
    def __init__(self, test_data_path="./processed_data/test.csv"):
        self.test_data_path = test_data_path
        self.results = {}
        
    def load_test_data(self):
        """Load test data"""
        if not os.path.exists(self.test_data_path):
            print(f"Test data not found: {self.test_data_path}")
            return None
        
        df = pd.read_csv(self.test_data_path)
        print(f"Loaded {len(df)} test samples")
        return df
    
    def test_original_model(self, test_df):
        """Test the original feature-based model"""
        print("\nTesting original feature-based model...")
        
        try:
            # Initialize original model
            original_detector = StutterDetector()
            
            # Load model if it exists
            if os.path.exists("../best_stutter_model.pth"):
                original_detector.model = original_detector.model.__class__().to(original_detector.device)
                original_detector.model.load_state_dict(torch.load("../best_stutter_model.pth", map_location=original_detector.device))
                print("✅ Loaded original model")
            else:
                print("❌ Original model not found. Skipping original model test.")
                return None
            
            # Test predictions
            predictions = []
            true_labels = []
            
            for _, row in test_df.iterrows():
                audio_path = row['audio_path']
                true_label = row['label']
                
                result = original_detector.predict_single_audio(audio_path)
                if result:
                    predictions.append(result['predicted_class'])
                    true_labels.append(true_label)
            
            if predictions:
                return {
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'model_name': 'Original Feature-Based'
                }
            
        except Exception as e:
            print(f"Error testing original model: {e}")
        
        return None
    
    def test_wav2vec2_model(self, test_df):
        """Test the Wav2Vec2 model"""
        print("\nTesting Wav2Vec2 model...")
        
        try:
            # Import Wav2Vec2 inference
            from inference_wav2vec2 import Wav2Vec2StutterInference
            
            # Initialize Wav2Vec2 model
            wav2vec2_inference = Wav2Vec2StutterInference()
            
            # Test predictions
            predictions = []
            true_labels = []
            
            for _, row in test_df.iterrows():
                audio_path = row['audio_path']
                true_label = row['label']
                
                result = wav2vec2_inference.predict_single(audio_path)
                if result:
                    predictions.append(result['predicted_class'])
                    true_labels.append(true_label)
            
            if predictions:
                return {
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'model_name': 'Wav2Vec2 Fine-tuned'
                }
            
        except Exception as e:
            print(f"Error testing Wav2Vec2 model: {e}")
        
        return None
    
    def calculate_metrics(self, predictions, true_labels, model_name):
        """Calculate performance metrics"""
        if not predictions or not true_labels:
            return None
        
        # Classification report
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Calculate accuracy
        accuracy = (np.array(predictions) == np.array(true_labels)).mean()
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return metrics
    
    def plot_comparison(self, results):
        """Plot comparison of model performances"""
        if not results:
            print("No results to plot")
            return
        
        # Prepare data for plotting
        model_names = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = {metric: [] for metric in metrics}
        
        for result in results:
            if result:
                model_names.append(result['model_name'])
                for metric in metrics:
                    metric_values[metric].append(result[metric])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            axes[row, col].bar(model_names, metric_values[metric])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(metric_values[metric]):
                axes[row, col].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Comparison plot saved: model_comparison.png")
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models"""
        if not results:
            return
        
        n_models = len([r for r in results if r])
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        
        for i, result in enumerate(results):
            if result:
                cm = np.array(result['confusion_matrix'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names,
                           ax=axes[i])
                axes[i].set_title(f'{result["model_name"]}\nConfusion Matrix')
                axes[i].set_ylabel('True Label')
                axes[i].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrices saved: confusion_matrices_comparison.png")
    
    def save_comparison_results(self, results):
        """Save comparison results to JSON"""
        if not results:
            return
        
        # Clean up results for JSON serialization
        clean_results = []
        for result in results:
            if result:
                clean_result = {
                    'model_name': result['model_name'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'confusion_matrix': result['confusion_matrix']
                }
                clean_results.append(clean_result)
        
        with open('model_comparison_results.json', 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print("Comparison results saved: model_comparison_results.json")
    
    def print_comparison_summary(self, results):
        """Print comparison summary"""
        if not results:
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for result in results:
            if result:
                comparison_data.append({
                    'Model': result['model_name'],
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'Precision': f"{result['precision']:.3f}",
                    'Recall': f"{result['recall']:.3f}",
                    'F1-Score': f"{result['f1_score']:.3f}"
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # Find best model
            best_model = max(results, key=lambda x: x['f1_score'] if x else 0)
            if best_model:
                print(f"\n🏆 Best Model: {best_model['model_name']}")
                print(f"   F1-Score: {best_model['f1_score']:.3f}")
                print(f"   Accuracy: {best_model['accuracy']:.3f}")
        
        print("="*80)
    
    def run_comparison(self):
        """Run complete model comparison"""
        print("🔍 MODEL COMPARISON: Original vs Wav2Vec2")
        print("="*60)
        
        # Load test data
        test_df = self.load_test_data()
        if test_df is None:
            return
        
        # Test original model
        original_results = self.test_original_model(test_df)
        
        # Test Wav2Vec2 model
        wav2vec2_results = self.test_wav2vec2_model(test_df)
        
        # Calculate metrics
        results = []
        
        if original_results:
            original_metrics = self.calculate_metrics(
                original_results['predictions'],
                original_results['true_labels'],
                original_results['model_name']
            )
            if original_metrics:
                results.append(original_metrics)
        
        if wav2vec2_results:
            wav2vec2_metrics = self.calculate_metrics(
                wav2vec2_results['predictions'],
                wav2vec2_results['true_labels'],
                wav2vec2_results['model_name']
            )
            if wav2vec2_metrics:
                results.append(wav2vec2_metrics)
        
        # Generate comparison
        if results:
            self.print_comparison_summary(results)
            self.plot_comparison(results)
            self.plot_confusion_matrices(results)
            self.save_comparison_results(results)
            
            print("\n✅ Comparison completed successfully!")
            print("Generated files:")
            print("- model_comparison.png")
            print("- confusion_matrices_comparison.png")
            print("- model_comparison_results.json")
        else:
            print("\n❌ No results to compare. Check if models are available.")

def main():
    """Main function"""
    comparator = ModelComparator()
    comparator.run_comparison()

if __name__ == "__main__":
    main() 