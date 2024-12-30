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

## Introduction
This project aims to develop an emotion recognition system from speech using Mel Spectrograms and Convolutional Neural Networks (CNNs). Speech signals often carry subtle emotional cues, which can be captured through time-frequency representations like Mel Spectrograms. Leveraging CNNs, the project seeks to identify these cues and classify emotional states effectively.

---

## Objectives
- Analyze and preprocess the Acted Emotional Speech Dynamic Database.
- Convert audio signals into Mel Spectrograms to extract features.
- Design and train CNN models for emotion classification.
- Evaluate model performance using metrics such as accuracy, precision, and recall.
- Document findings and insights for future research.

---

## Dataset
The project utilizes the [Acted Emotional Speech Dynamic Database](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database), which contains recordings of emotional expressions such as happiness, sadness, anger, and more. 

### Dataset Overview
- **Number of Samples**: [Insert total number]
- **Emotions Included**: [List emotions]
- **Audio Format**: [e.g., WAV, MP3]

### Data Distribution
- Distribution of samples across emotions:  
  ![Sample Distribution Chart](path/to/chart.png)

---

## Methodology

### Audio Preprocessing
- **Noise Reduction**: Applied [method/technique].
- **Normalization**: Scaled audio signals to a uniform range.

### Mel Spectrogram Conversion
- Parameters used:
  - Sample Rate: [e.g., 16 kHz]
  - FFT Size: [e.g., 2048]
  - Hop Length: [e.g., 512]
  - Number of Mel Bands: [e.g., 128]
- Spectrogram Example:  
  ![Mel Spectrogram](path/to/spectrogram.png)

### CNN Model Design
- **Model Architecture**: [e.g., Conv2D layers with batch normalization, pooling, and dense layers].
- **Activation Functions**: [e.g., ReLU, Softmax].
- **Loss Function**: [e.g., Categorical Crossentropy].
- **Optimizer**: [e.g., Adam].

### Training and Validation
- **Training Set Size**: [e.g., 70% of data]
- **Validation Set Size**: [e.g., 20% of data]
- **Testing Set Size**: [e.g., 10% of data]
- **Epochs**: [e.g., 50]
- **Batch Size**: [e.g., 32]

---

## Results
### Performance Metrics
- **Accuracy**: [Insert value]
- **Precision**: [Insert value]
- **Recall**: [Insert value]

### Confusion Matrix
![Confusion Matrix](path/to/confusion_matrix.png)

---

## Challenges and Solutions
- **Challenge**: Imbalanced data across emotional categories.
  - **Solution**: Used data augmentation techniques to balance the dataset.
- **Challenge**: Overfitting during training.
  - **Solution**: Applied dropout layers and early stopping.

---

## Conclusions
The project successfully developed a CNN-based system for emotion recognition from speech. The best-performing model achieved an accuracy of [Insert value], demonstrating the potential of Mel Spectrograms in capturing emotional features.

---

## Future Work
- Experiment with other feature extraction methods, such as MFCCs.
- Explore transfer learning with pre-trained audio models.
- Increase the dataset size to improve generalizability.

---

## References
1. Acted Emotional Speech Dynamic Database: [Link](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database)
2. [Additional References Here]

---