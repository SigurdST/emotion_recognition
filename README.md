# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Overview

This project focuses on recognizing emotions from speech using **Mel Spectrograms** and **Convolutional Neural Networks (CNNs)**. The model is trained to classify audio recordings into various emotional categories using the **Acted Emotional Speech Dynamic Database**, a dataset of recordings featuring diverse emotional expressions.

## Objectives

- Analyze and preprocess speech data to generate Mel Spectrograms.
- Develop CNN architectures optimized for emotion classification.
- Train, validate, and optimize models for robust performance.
- Document methodologies, results, and insights for reproducibility.

---

## Dataset

- **Source**: [Acted Emotional Speech Dynamic Database on DagsHub](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database)
- **Description**: The dataset contains acted speech recordings labeled with emotional states such as happy, sad, angry, etc.
- **Key Features**:
  - Multiple speakers with varying speech styles.
  - Diverse emotional expressions.
  - Metadata for labels and recording details.

---

## Workflow

### 1. Dataset Analysis
- Explore the dataset structure and distribution of emotions.
- Visualize audio characteristics like duration and sampling rates.

### 2. Audio Preprocessing
- Apply noise reduction and normalization for consistent audio quality.
- Convert audio recordings into Mel Spectrograms using `librosa`.

### 3. Model Development
- Build CNN architectures designed to process spectrogram inputs.
- Experiment with different configurations to capture emotion-specific patterns.

### 4. Training and Validation
- Split data into training, validation, and test sets.
- Use stratified k-fold validation for robust performance evaluation.

### 5. Performance Evaluation
- Assess models using metrics like accuracy, precision, recall, and F1-score.
- Generate confusion matrices to visualize classification results.

### 6. Optimization
- Apply hyperparameter tuning, data augmentation, and architectural improvements to enhance accuracy.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or above
- Libraries: `librosa`, `numpy`, `pandas`, `matplotlib`, `PyTorch` or `TensorFlow`

### Steps to Run
1. Clone this repository:
   ```bash
   git clone <repository-link>
   cd <repository-name>