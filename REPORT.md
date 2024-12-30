# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
   - [Audio Preprocessing](#audio-preprocessing)
   - [Mel Spectrogram Conversion](#mel-spectrogram-conversion)
   - [CNN Model Design](#cnn-model-design)
   - [Training and Validation](#training-and-validation)
5. [Results](#results)
6. [Challenges and Solutions](#challenges-and-solutions)
7. [Conclusions](#conclusions)
8. [Future Work](#future-work)
9. [References](#references)

---

## Methodology

### Audio Preprocessing
To preprocess the audio files, noise reduction and normalization were applied using Python. Below is a snippet of the preprocessing code:

```python
import librosa
import numpy as np

def preprocess_audio(file_path):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # Noise reduction
    noise_profile = np.mean(audio)
    audio_denoised = audio - noise_profile
    
    # Normalization
    audio_normalized = librosa.util.normalize(audio_denoised)
    
    return audio_normalized, sample_rate