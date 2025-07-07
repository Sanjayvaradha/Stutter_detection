#!/usr/bin/env python3
"""
Data preparation script for Wav2Vec2 fine-tuning
Processes the FluencyBank dataset for transformer-based stutter detection
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class StutterDataPreparator:
    """Prepare data for Wav2Vec2 fine-tuning"""
    
    def __init__(self, audio_dir="../clips/clips", labels_file="../fluencybank_labels.csv"):
        self.audio_dir = audio_dir
        self.labels_file = labels_file
        self.sample_rate = 16000  # Wav2Vec2 expects 16kHz
        
    def load_and_process_labels(self):
        """Load CSV and process labels according to stutter percentage"""
        print("Loading and processing labels...")
        
        # Load CSV
        df = pd.read_csv(self.labels_file)
        df = df.dropna()
        
        # Prepare data lists
        audio_paths = []
        labels = []
        stutter_percentages = []
        episode_ids = []
        clip_ids = []
        
        print("Processing audio files and calculating stutter percentages...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ep_id = str(row['EpId']).strip()
            clip_id = str(row['ClipId']).strip()
            
            # Create audio file path
            audio_filename = f"FluencyBank_{ep_id}_{clip_id}.wav"
            audio_path = os.path.join(self.audio_dir, audio_filename)
            
            # Check if audio file exists
            if os.path.exists(audio_path):
                # Calculate stutter percentage
                stutter_types = ['Prolongation', 'Block', 'SoundRep', 'WordRep']
                total_stutters = sum(row[stutter_type] for stutter_type in stutter_types)
                total_words = row['NoStutteredWords']
                
                if total_words > 0:
                    stutter_percentage = (total_stutters / total_words) * 100
                    
                    # Classify based on percentage
                    if stutter_percentage < 3:
                        final_label = 0  # No stutter
                    elif stutter_percentage <= 5:
                        final_label = 1  # At risk
                    else:
                        final_label = 2  # Stutters
                    
                    audio_paths.append(audio_path)
                    labels.append(final_label)
                    stutter_percentages.append(stutter_percentage)
                    episode_ids.append(ep_id)
                    clip_ids.append(clip_id)
        
        print(f"Found {len(audio_paths)} valid audio-label pairs")
        
        # Create DataFrame
        data_df = pd.DataFrame({
            'audio_path': audio_paths,
            'label': labels,
            'stutter_percentage': stutter_percentages,
            'episode_id': episode_ids,
            'clip_id': clip_ids
        })
        
        # Print class distribution
        print("\nClass Distribution:")
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        for i, class_name in enumerate(class_names):
            count = (data_df['label'] == i).sum()
            percentage = (count / len(data_df)) * 100
            print(f"{class_name}: {count} samples ({percentage:.1f}%)")
        
        return data_df
    
    def split_data(self, data_df, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets"""
        print(f"\nSplitting data (test_size={test_size}, val_size={val_size})...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            data_df, test_size=test_size, random_state=random_state, 
            stratify=data_df['label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=random_state,
            stratify=train_val_df['label']
        )
        
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def create_huggingface_datasets(self, train_df, val_df, test_df):
        """Create HuggingFace Dataset objects"""
        print("\nCreating HuggingFace datasets...")
        
        # Convert DataFrames to HuggingFace Datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        print("Dataset structure:")
        print(dataset_dict)
        
        return dataset_dict
    
    def verify_audio_files(self, dataset_dict):
        """Verify that all audio files exist and are readable"""
        print("\nVerifying audio files...")
        
        all_datasets = ['train', 'validation', 'test']
        valid_files = []
        invalid_files = []
        
        for split_name in all_datasets:
            dataset = dataset_dict[split_name]
            print(f"\nChecking {split_name} split...")
            
            for i, example in enumerate(tqdm(dataset, desc=f"Verifying {split_name}")):
                audio_path = example['audio_path']
                
                try:
                    # Try to load audio file
                    audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=1.0)
                    if len(audio) > 0:
                        valid_files.append(audio_path)
                    else:
                        invalid_files.append(audio_path)
                except Exception as e:
                    invalid_files.append(audio_path)
                    print(f"Error loading {audio_path}: {e}")
        
        print(f"\nVerification complete:")
        print(f"Valid files: {len(valid_files)}")
        print(f"Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            print("\nInvalid files:")
            for file in invalid_files[:10]:  # Show first 10
                print(f"  {file}")
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more")
        
        return len(invalid_files) == 0
    
    def save_processed_data(self, dataset_dict, output_dir="./processed_data"):
        """Save processed datasets"""
        print(f"\nSaving processed data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV files for inspection
        for split_name, dataset in dataset_dict.items():
            df = dataset.to_pandas()
            csv_path = os.path.join(output_dir, f"{split_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {split_name} split: {csv_path}")
        
        # Save dataset dict
        dataset_dict.save_to_disk(os.path.join(output_dir, "dataset_dict"))
        print(f"Saved dataset dict: {os.path.join(output_dir, 'dataset_dict')}")
        
        # Save metadata
        metadata = {
            'total_samples': sum(len(dataset) for dataset in dataset_dict.values()),
            'train_samples': len(dataset_dict['train']),
            'validation_samples': len(dataset_dict['validation']),
            'test_samples': len(dataset_dict['test']),
            'sample_rate': self.sample_rate,
            'class_names': ['No Stutter', 'At Risk', 'Stutters']
        }
        
        import json
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {os.path.join(output_dir, 'metadata.json')}")
    
    def run_complete_preparation(self, output_dir="./processed_data"):
        """Run complete data preparation pipeline"""
        print("=== Stutter Data Preparation for Wav2Vec2 ===")
        
        # Step 1: Load and process labels
        data_df = self.load_and_process_labels()
        
        # Step 2: Split data
        train_df, val_df, test_df = self.split_data(data_df)
        
        # Step 3: Create HuggingFace datasets
        dataset_dict = self.create_huggingface_datasets(train_df, val_df, test_df)
        
        # Step 4: Verify audio files
        all_valid = self.verify_audio_files(dataset_dict)
        
        if not all_valid:
            print("Warning: Some audio files are invalid!")
        
        # Step 5: Save processed data
        self.save_processed_data(dataset_dict, output_dir)
        
        print("\n=== Data Preparation Complete ===")
        return dataset_dict

def main():
    """Main function"""
    preparator = StutterDataPreparator()
    dataset_dict = preparator.run_complete_preparation()
    
    print("\nNext steps:")
    print("1. Run: python train_wav2vec2.py")
    print("2. Run: python inference_wav2vec2.py")

if __name__ == "__main__":
    main() 