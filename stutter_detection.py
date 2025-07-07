import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class StutterDataset(Dataset):
    """Custom Dataset for stutter detection"""
    def __init__(self, audio_paths, labels, transform=None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            features = self.extract_features(audio, sr)
            
            if self.transform:
                features = self.transform(features)
            
            return torch.FloatTensor(features), torch.FloatTensor(label)
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zero features if audio loading fails
            return torch.zeros(128), torch.FloatTensor(label)
    
    def extract_features(self, audio, sr):
        """Extract audio features"""
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Root mean square energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Combine all features
        features = []
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
        features.extend([np.mean(rms), np.std(rms)])
        features.extend(np.mean(chroma, axis=1))
        
        # Pad or truncate to fixed length
        if len(features) < 128:
            features.extend([0] * (128 - len(features)))
        else:
            features = features[:128]
        
        return features

class StutterClassifier(nn.Module):
    """Neural network for stutter classification"""
    def __init__(self, input_size=128, hidden_size=256, num_classes=3):
        super(StutterClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 4, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

class StutterDetector:
    """Main class for stutter detection"""
    def __init__(self, audio_dir="clips/clips", labels_file="fluencybank_labels.csv"):
        self.audio_dir = audio_dir
        self.labels_file = labels_file
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading labels...")
        df = pd.read_csv(self.labels_file)
        
        # Clean the data
        df = df.dropna()
        
        # Create audio file paths
        audio_paths = []
        labels = []
        
        print("Processing audio files...")
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
        
        print(f"Found {len(audio_paths)} valid audio-label pairs")
        
        # Split the data
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            audio_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    
    def create_data_loaders(self, train_paths, val_paths, test_paths, 
                           train_labels, val_labels, test_labels, batch_size=32):
        """Create data loaders for training"""
        train_dataset = StutterDataset(train_paths, train_labels)
        val_dataset = StutterDataset(val_paths, val_labels)
        test_dataset = StutterDataset(test_paths, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Train the stutter detection model"""
        print("Initializing model...")
        self.model = StutterClassifier().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.long().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.long().to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_stutter_model.pth')
                print("Saved best model!")
            
            scheduler.step(avg_val_loss)
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png')
        plt.close()
    
    def evaluate_model(self, test_loader):
        """Evaluate the trained model"""
        if self.model is None:
            print("Loading best model...")
            self.model = StutterClassifier().to(self.device)
            self.model.load_state_dict(torch.load('best_stutter_model.pth'))
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(test_loader, desc="Evaluating"):
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        # Print classification report
        class_names = ['No Stutter', 'At Risk', 'Stutters']
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions, class_names)
        
        return all_predictions, all_labels
    
    def plot_confusion_matrix(self, true_labels, predicted_labels, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def predict_single_audio(self, audio_path):
        """Predict stutter classification for a single audio file"""
        if self.model is None:
            print("Loading best model...")
            self.model = StutterClassifier().to(self.device)
            self.model.load_state_dict(torch.load('best_stutter_model.pth'))
        
        self.model.eval()
        
        # Load and preprocess audio
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            dataset = StutterDataset([audio_path], [0])  # Dummy label
            features, _ = dataset[0]
            features = features.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = outputs.cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
            
            class_names = ['No Stutter', 'At Risk', 'Stutters']
            confidence = probabilities[predicted_class]
            
            return {
                'predicted_class': predicted_class,
                'class_name': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities
            }
        
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete stutter detection pipeline"""
        print("=== Stutter Detection Pipeline ===")
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = self.load_and_preprocess_data()
        
        # Step 2: Create data loaders
        print("\n2. Creating data loaders...")
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
        )
        
        # Step 3: Train model
        print("\n3. Training model...")
        train_losses, val_losses = self.train_model(train_loader, val_loader)
        
        # Step 4: Evaluate model
        print("\n4. Evaluating model...")
        predictions, true_labels = self.evaluate_model(test_loader)
        
        print("\n=== Pipeline Complete ===")
        print("Model saved as 'best_stutter_model.pth'")
        print("Training curves saved as 'training_curves.png'")
        print("Confusion matrix saved as 'confusion_matrix.png'")

def main():
    """Main function to run the stutter detection"""
    detector = StutterDetector()
    detector.run_complete_pipeline()

if __name__ == "__main__":
    main() 