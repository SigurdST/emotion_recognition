# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Table of Contents
1. [Project Presentation](#project-presentation)
2. [Data Preparation](#data-preparation)
   - [Data Exploration](#data-exploration)
   - [Data Processing](#data-processing)
3. [Convolutional Neural Networks](#convolutional-neural-networks)
   - [CNNs with Batch Size of 1](#cnns-with-batch-size-of-1)
   - [CNNs with Resized Spectrograms](#cnns-with-resized-spectrograms)
   - [CNNs on Artificial Data](#cnns-on-artificial-data)
4. [Conclusions](#conclusions)
5. [References](#references)

---

## Project Presentation

This project focuses on developing an emotion recognition system from speech using Mel Spectrograms and Convolutional Neural Networks (CNNs). The dataset used is the [Acted Emotional Speech Dynamic Database (AESDD)](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database), which is publicly available on DagsHub.

The workflow begins with an analysis of the dataset, followed by transforming the audio recordings into numerical representations using the Mel Spectrogram technique. These spectrograms serve as input features for CNNs, which will be designed and trained to classify emotions.

Various optimization techniques will be applied to enhance the performance of the CNN models, and different approaches will be compared to identify the most effective methods for emotion recognition.

---

## Data Preparation

### Data Exploration

The AESDD consists of 605 audio files, categorized into five emotions: angry, disgust, fear, happy, and sad.

I began by importing the audio files and used the `librosa` package to analyze their characteristics and visualize their waveforms.

![waveform](plots/waveform.png)

A waveform represents the variation of an audio signal over time, with **time** on the x-axis and **amplitude** (ranging between -1 and 1) on the y-axis. It visually displays how the sound's pressure changes over time, allowing me to observe characteristics like volume and temporal structure of the audio. Amplitude refers to the magnitude of the audio signal, representing the intensity or loudness of the sound at a given point in time.

I also created a histogram of the audio file durations to gain an overall understanding of their distribution.

![hist_duration](plots/hist_duration.png)

I noticed that one audio file had a duration of 0.0 seconds, so I removed it from the dataset.

Next, I compared the waveforms of one audio file from each emotion, but it was challenging to draw clear conclusions from the visualizations. Additionally, I compared the durations across different emotions and found them to be largely similar.

### Data Processing

**Noise Reduction**

To process our data, we first apply noise reduction using the `noisereduce` package. The `nr.reduce_noise` function from this library reduces background noise in audio signals while preserving the primary sound, such as speech or music. It achieves this by analyzing a noise profile to estimate the characteristics of the background noise, which is then subtracted from the audio signal's frequency spectrum. This technique enhances audio clarity by attenuating noise-dominated frequencies while retaining the integrity of the desired signal.

**Mel Spectrogram**

A Mel Spectrogram is a time-frequency representation of audio where the frequency axis is scaled according to the **Mel scale**, which approximates human auditory perception. Here's a breakdown of how it works mathematically:

1. *Short-Time Fourier Transform (STFT)*:
   The audio signal $x(t)$ is divided into overlapping frames, and the Fourier Transform is applied to each frame to obtain the frequency spectrum:
   
   $$ X(f, t) = \text{STFT}(x(t))$$
   
   where $X(f, t)$ represents the frequency components at a specific time.

3. *Power Spectrogram*:
   The magnitude of the STFT is squared to calculate the power spectrum:

   $$
   P(f, t) = |X(f, t)|^2
   $$

4. *Mapping to Mel Scale*:
   Frequencies are converted to the Mel scale using a triangular filter bank. The Mel scale is defined as:

   $$
   m(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)
   $$
   Each filter in the bank sums the power within its frequency range, effectively smoothing the spectrum.

5. *Mel Filter Bank Application*:
   The power spectrogram is multiplied by the Mel filter bank to map the linear frequency scale to the Mel scale:

   $$
   M(m, t) = \sum_{f} P(f, t) \cdot H_m(f)
   $$

   where $H_m(f)$ represents the filter weights for the $m$-th Mel filter.

6. *Logarithmic Compression*:
   To mimic the human perception of sound intensity, a logarithmic transformation is applied:

   $$
   \text{Mel Spectrogram}(m, t) = \log\left(M(m, t) + \epsilon\right)
   $$
   where $\epsilon$ is a small value to avoid logarithm of zero.
