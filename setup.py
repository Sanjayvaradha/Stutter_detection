#!/usr/bin/env python3
"""
Setup script for Stutter Detection System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    return True

def verify_dataset():
    """Verify dataset structure"""
    print("\nVerifying dataset structure...")
    
    # Check if labels file exists
    if not os.path.exists("fluencybank_labels.csv"):
        print("❌ fluencybank_labels.csv not found!")
        return False
    
    # Check if audio directory exists
    if not os.path.exists("clips/clips"):
        print("❌ clips/clips directory not found!")
        return False
    
    # Count audio files
    try:
        audio_files = [f for f in os.listdir("clips/clips") if f.endswith('.wav')]
        print(f"✅ Found {len(audio_files)} audio files")
    except Exception as e:
        print(f"❌ Error reading audio directory: {e}")
        return False
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    
    required_modules = [
        'torch', 'librosa', 'pandas', 'numpy', 'sklearn', 
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - not installed")
            return False
    
    return True

def main():
    """Main setup function"""
    print("=== Stutter Detection System Setup ===\n")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Some required modules are missing. Please install them manually.")
        return
    
    # Verify dataset
    if not verify_dataset():
        print("\n❌ Dataset structure is incorrect. Please check the README for correct structure.")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run:")
    print("  python stutter_detection.py  # Train the model")
    print("  python inference.py --audio path/to/audio.wav  # Test inference")

if __name__ == "__main__":
    main() 