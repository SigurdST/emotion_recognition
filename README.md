# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Overview

This project focuses on developing an emotion recognition system from speech using Mel Spectrograms and Convolutional Neural Networks (CNNs). The dataset used is the [Acted Emotional Speech Dynamic Database (AESDD)](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database), which contains audio files categorized into five emotions: angry, disgust, fear, happy, and sad.

## Objectives

1. Transform raw audio files into numerical representations using Mel Spectrograms.
2. Train CNNs to classify emotions from these spectrograms.
3. Address challenges such as:
   - Variable input shapes.
   - Small dataset size.
4. Compare performance using original, resized, and augmented datasets.

## Key Findings

1. The best results were obtained using the original dataset with a batch size of 1, achieving an accuracy of **74.38%**.
2. Resized spectrograms resulted in lower accuracy, likely due to loss of crucial information during resizing.
3. Artificially augmented data achieved accuracy similar to the original dataset, but with possible information loss during augmentation.

## Repository Contents

- `notebook.ipynb`: Code for data preprocessing, CNN training, and evaluation.
- `plots/`: Visualizations of spectrograms, loss convergence, and results.
- `REPORT.md`: In-depth details of the project workflow, methods, challenges, and conclusions.

## Quick Start

1. Clone the repository:  
   ```bash
   git clone https://github.com/SigurdST/emotion_recognition.git
   cd emotion_recognition# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Overview

This project focuses on developing an emotion recognition system from speech using Mel Spectrograms and Convolutional Neural Networks (CNNs). The dataset used is the [Acted Emotional Speech Dynamic Database (AESDD)](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database), which contains audio files categorized into five emotions: angry, disgust, fear, happy, and sad.

## Objectives

1. Transform raw audio files into numerical representations using Mel Spectrograms.
2. Train CNNs to classify emotions from these spectrograms.
3. Address challenges such as:
   - Variable input shapes.
   - Small dataset size.
4. Compare performance using original, resized, and augmented datasets.

## Key Findings

1. The best results were obtained using the original dataset with a batch size of 1, achieving an accuracy of **74.38%**.
2. Resized spectrograms resulted in lower accuracy, likely due to loss of crucial information during resizing.
3. Artificially augmented data achieved accuracy similar to the original dataset, but with possible information loss during augmentation.

## Repository Contents

- `notebook.ipynb`: Code for data preprocessing, CNN training, and evaluation.
- `REPORT.md`: In-depth details of the project workflow, methods, challenges, and conclusions.

## Quick Start

1. Clone the repository:  
   ```bash
   git clone https://github.com/SigurdST/emotion_recognition.git
   cd emotion_recognition# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Overview

This project focuses on developing an emotion recognition system from speech using Mel Spectrograms and Convolutional Neural Networks (CNNs). The dataset used is the [Acted Emotional Speech Dynamic Database (AESDD)](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database), which contains audio files categorized into five emotions: angry, disgust, fear, happy, and sad.

## Objectives

1. Transform raw audio files into numerical representations using Mel Spectrograms.
2. Train CNNs to classify emotions from these spectrograms.
3. Address challenges such as:
   - Variable input shapes.
   - Small dataset size.
4. Compare performance using original data with batch of size 1, resized data, and augmented datasets.

## Key Findings

1. The best results were obtained using the original dataset with a batch size of 1, achieving an accuracy of **74.38%**.
2. Resized spectrograms resulted in lower accuracy, likely due to loss of crucial information during resizing.
3. Artificially augmented data achieved accuracy similar to the original dataset, but with possible information loss during augmentation.

## Repository Contents

- `notebook.ipynb`: Code for data preprocessing, CNN training, and evaluation.
- `plots/`: Visualizations of spectrograms, loss convergence, and results.
- `REPORT.md`: In-depth details of the project workflow, methods, challenges, and conclusions.

## Quick Start

1. Clone the repository:  
   ```bash
   git clone https://github.com/SigurdST/emotion_recognition.git
   cd emotion_recognition

2. Explore `notebook.ipynb` to review all the code implementation and processes, and `REPORT.md` for detailed explanations, results, and insights derived from the project.