#!/usr/bin/env python3
"""
Wav2Vec2 fine-tuning script for stutter detection
Uses pre-trained Wav2Vec2 model and fine-tunes it for 3-class stutter classification
"""

import os
import torch
import numpy as np
import librosa
from datasets import load_from_disk, Dataset
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

class Wav2Vec2StutterTrainer:
    """Wav2Vec2 trainer for stutter detection"""
    
    def __init__(self, 
                 model_name="facebook/wav2vec2-base",
                 data_dir="./processed_data",
                 output_dir="./wav2vec2_stutter_model",
                 sample_rate=16000):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_processed_data(self):
        """Load the processed dataset"""
        print("Loading processed dataset...")
        
        try:
            # Try to load from disk
            dataset_dict = load_from_disk(os.path.join(self.data_dir, "dataset_dict"))
            print("Loaded dataset from disk")
        except:
            print("Dataset not found on disk. Please run data_preparation.py first.")
            return None
        
        # Load metadata
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata: {self.metadata}")
        
        return dataset_dict
    
    def load_model_and_processor(self):
        """Load pre-trained Wav2Vec2 model and processor"""
        print(f"Loading pre-trained model: {self.model_name}")
        
        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        
        # Load model
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,  # 3 classes: No Stutter, At Risk, Stutters
            gradient_checkpointing=True,  # Save memory
            use_cache=False  # Save memory during training
        )
        
        # Move to device
        self.model.to(self.device)
        
        print(f"Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def preprocess_function(self, batch):
        """Preprocess audio data for Wav2Vec2"""
        try:
            # Load audio
            audio, sr = librosa.load(batch["audio_path"], sr=self.sample_rate)
            
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Pad or truncate to 30 seconds (Wav2Vec2 can handle variable length)
            max_length = 30 * self.sample_rate
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                # Pad with zeros
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            
            # Process with Wav2Vec2 processor
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            batch["input_values"] = inputs.input_values[0]
            batch["labels"] = int(batch["label"])
            
        except Exception as e:
            print(f"Error preprocessing {batch['audio_path']}: {e}")
            # Return dummy data if preprocessing fails
            batch["input_values"] = torch.zeros(16000 * 30)  # 30 seconds of zeros
            batch["labels"] = int(batch["label"])
        
        return batch
    
    def data_collator(self, features):
        """Custom data collator for Wav2Vec2"""
        input_values = torch.stack([f["input_values"] for f in features])
        labels = torch.tensor([f["labels"] for f in features])
        
        return {
            "input_values": input_values,
            "labels": labels
        }
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = (predictions == labels).astype(np.float32).mean()
        
        # Classification report
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1": report['weighted avg']['f1-score']
        }
    
    def plot_confusion_matrix(self, predictions, labels, save_path):
        """Plot and save confusion matrix"""
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Wav2Vec2 Stutter Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {save_path}")
    
    def setup_training_arguments(self, num_epochs=10, batch_size=4, learning_rate=1e-4):
        """Setup training arguments"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_num_workers=4,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to=None,  # Disable wandb for now
            save_total_limit=3,  # Keep only 3 best checkpoints
            remove_unused_columns=False,
        )
        
        return training_args
    
    def train_model(self, dataset_dict, num_epochs=10, batch_size=4, learning_rate=1e-4):
        """Train the Wav2Vec2 model"""
        print("Setting up training...")
        
        # Preprocess datasets
        print("Preprocessing training data...")
        train_dataset = dataset_dict["train"].map(
            self.preprocess_function,
            remove_columns=dataset_dict["train"].column_names,
            desc="Preprocessing train"
        )
        
        print("Preprocessing validation data...")
        val_dataset = dataset_dict["validation"].map(
            self.preprocess_function,
            remove_columns=dataset_dict["validation"].column_names,
            desc="Preprocessing validation"
        )
        
        # Setup training arguments
        training_args = self.setup_training_arguments(num_epochs, batch_size, learning_rate)
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Save training results
        with open(os.path.join(self.output_dir, "training_results.json"), 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        print(f"Training completed! Model saved to: {self.output_dir}")
        
        return trainer, train_result
    
    def evaluate_model(self, trainer, dataset_dict):
        """Evaluate the trained model on test set"""
        print("Evaluating model on test set...")
        
        # Preprocess test data
        test_dataset = dataset_dict["test"].map(
            self.preprocess_function,
            remove_columns=dataset_dict["test"].column_names,
            desc="Preprocessing test"
        )
        
        # Evaluate
        eval_results = trainer.evaluate(test_dataset)
        
        # Get predictions for confusion matrix
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Plot confusion matrix
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(pred_labels, true_labels, cm_path)
        
        # Save evaluation results
        eval_results['predictions'] = pred_labels.tolist()
        eval_results['true_labels'] = true_labels.tolist()
        
        with open(os.path.join(self.output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print("Evaluation Results:")
        for key, value in eval_results.items():
            if key not in ['predictions', 'true_labels']:
                print(f"{key}: {value:.4f}")
        
        return eval_results
    
    def run_complete_training(self, num_epochs=10, batch_size=4, learning_rate=1e-4):
        """Run complete training pipeline"""
        print("=== Wav2Vec2 Stutter Detection Training ===")
        
        # Step 1: Load data
        dataset_dict = self.load_processed_data()
        if dataset_dict is None:
            return
        
        # Step 2: Load model and processor
        self.load_model_and_processor()
        
        # Step 3: Train model
        trainer, train_result = self.train_model(dataset_dict, num_epochs, batch_size, learning_rate)
        
        # Step 4: Evaluate model
        eval_results = self.evaluate_model(trainer, dataset_dict)
        
        print("\n=== Training Complete ===")
        print(f"Model saved to: {self.output_dir}")
        print(f"Best model checkpoint: {self.output_dir}")
        print(f"Training results: {os.path.join(self.output_dir, 'training_results.json')}")
        print(f"Evaluation results: {os.path.join(self.output_dir, 'evaluation_results.json')}")
        print(f"Confusion matrix: {os.path.join(self.output_dir, 'confusion_matrix.png')}")
        
        return trainer, eval_results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wav2Vec2 Stutter Detection Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base', 
                       help='Pre-trained model name')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Wav2Vec2StutterTrainer(model_name=args.model)
    
    # Run training
    trainer.run_complete_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == "__main__":
    main() 