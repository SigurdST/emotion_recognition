# Emotion Recognition from Speech using Mel Spectrograms and CNNs

## Table of Contents
1. [Project Presentation](#project-presentation)
2. [Data Preparation](#data-preparation)
   - [Data Exploration](#data-exploration)
   - [Data Processing](#data-processing)
3. [Convolutional Neural Networks](#convolutional-neural-networks)
   - [Presentation](#presentation)
   - [CNNs with Batch Size of 1](#cnns-with-batch-size-of-1)
   - [CNNs with Resized Spectrograms](#cnns-with-resized-spectrograms)
   - [CNNs on Artificial Data](#cnns-on-artificial-data)
4. [Conclusions](#conclusions)

---

# Project Presentation

This project focuses on developing an emotion recognition system from speech using Mel Spectrograms and Convolutional Neural Networks (CNNs). The dataset used is the [Acted Emotional Speech Dynamic Database (AESDD)](https://dagshub.com/kingabzpro/Acted-Emotional-Speech-Dynamic-Database), which is publicly available on DagsHub.

The workflow begins with an analysis of the dataset, followed by transforming the audio recordings into numerical representations using the Mel Spectrogram technique. These spectrograms serve as input features for CNNs, which will be designed and trained to classify emotions.

Various optimization techniques will be applied to enhance the performance of the CNN models, and different approaches will be compared to identify the most effective methods for emotion recognition.

All the code is in the `notebook.ipynb` file on my [GitHub repository](https://github.com/SigurdST/emotion_recognition).

---

# Data Preparation

## Data Exploration

The AESDD consists of 605 audio files, categorized into five emotions: angry, disgust, fear, happy, and sad. All the files have a sampling rate of 44100 Hz.

I began by importing the audio files and used the `librosa` package to analyze their characteristics and visualize their waveforms.

![waveform](plots/waveform.png)

A waveform represents the variation of an audio signal over time, with **time** on the x-axis and **amplitude** (ranging between -1 and 1) on the y-axis. It visually displays how the sound's pressure changes over time, allowing me to observe characteristics like volume and temporal structure of the audio. Amplitude refers to the magnitude of the audio signal, representing the intensity or loudness of the sound at a given point in time.

I also created a histogram of the audio file durations to gain an overall understanding of their distribution.

![hist_duration](plots/hist_duration.png)

I noticed that one audio file had a duration of 0.0 seconds, so I removed it from the dataset.

Next, I compared the waveforms of one audio file from each emotion, but it was challenging to draw clear conclusions from the visualizations. Additionally, I compared the durations across different emotions and found them to be largely similar.

## Data Processing

### Noise Reduction

To process our data, we first apply noise reduction using the `noisereduce` package. The `nr.reduce_noise` function from this library reduces background noise in audio signals while preserving the primary sound, such as speech or music. It achieves this by analyzing a noise profile to estimate the characteristics of the background noise, which is then subtracted from the audio signal's frequency spectrum. This technique enhances audio clarity by attenuating noise-dominated frequencies while retaining the integrity of the desired signal.

### Mel Spectrogram

A **Mel Spectrogram** is a time-frequency representation of audio, where the frequency axis is scaled according to the **Mel scale**, which approximates human auditory perception. This transformation provides numerical data that can be effectively used to train CNNs. Below is a breakdown of how it works mathematically:

#### 1. ***Short-Time Fourier Transform (STFT)***:  

   The audio signal $x(t)$ is divided into overlapping frames, and the Fourier Transform is applied to each frame to obtain the frequency spectrum:

$$
X(f, t) = \int_{-\infty}^{\infty} x(\tau) \cdot w(\tau - t) \cdot e^{-j 2 \pi f \tau} \, d\tau
$$

where:
- $x(\tau)$: the input signal as a function of time.
- $w(\tau - t)$: the window function centered around $t$, used to select a segment of the signal.
- $f$: the frequency at which the transform is computed.
- $e^{-j 2 \pi f \tau}$: the Fourier kernel, representing a complex sinusoid.

   The Fourier Transform is used to analyze the frequency content of a signal by converting it from the time domain to the frequency domain. This helps in understanding, processing, and filtering signals based on their frequency characteristics.


#### 2. ***Power Spectrogram***:  

   The magnitude of the STFT is squared to calculate the power spectrum:  

$$P(f, t) = |X(f, t)|^2$$  

   The power spectrum represents the distribution of power into frequency components of the signal over time. It helps identify which frequencies dominate the signal's energy at specific moments, making it essential in audio analysis, speech processing, and other signal processing tasks.

#### 3. ***Mapping to Mel Scale***:  

   Frequencies are converted to the Mel scale using a triangular filter bank. The Mel scale is defined as:  

$$m(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

   Each filter in the bank sums the power within its frequency range, effectively smoothing the spectrum.  

   The Mel scale approximates how humans perceive pitch, as it is designed to be more sensitive to lower frequencies and less sensitive to higher frequencies. This makes it useful for audio analysis tasks such as speech recognition and music processing.

#### 4. ***Mel Filter Bank Application***:

   The power spectrogram is multiplied by the Mel filter bank to map the linear frequency scale to the Mel scale:

$$M(m, t) = \sum_{f} P(f, t) \cdot H_m(f)$$

   where $H_m(f)$ represents the filter weights for the $m$-th Mel filter.

#### 5. ***Logarithmic Compression***: 

   To mimic the human perception of sound intensity, a logarithmic transformation is applied:  

$$\text{Mel Spectrogram}(m, t) = \log\left(M(m, t) + \epsilon\right)$$

   where $\epsilon$ is a small value to avoid the logarithm of zero.  

   The logarithmic transformation mimics the human perception of sound intensity because our hearing is more sensitive to relative changes in quiet sounds than in loud ones. This non-linear scaling reflects how humans perceive differences in sound levels, making it suitable for tasks like speech recognition and audio analysis.

### Application

To apply the Mel Spectrogram transformation to our data, we use the following functions from the `librosa` package:
- `feature.melspectrogram`: Computes the Mel Spectrogram.
- `power_to_db`: Performs logarithmic compression to mimic human sound perception.

In the `feature.melspectrogram` function, I chose the default values for both parameters: `n_mels=128`, which specifies 128 Mel bands for the spectrogram, and `n_fft=2048`, which defines the number of samples used in each Fourier Transform window to control the frequency resolution. The output is a $128*n$ matrices where :

$$
n = \lceil\frac{\text{sampling rate} \times \text{audio duration}}{\text{nfft}} \times 4\rceil
$$

Here, $n$ corresponds to the number of time frames in the spectrogram, and each element of the matrix represents the energy of a specific Mel band (row) at a specific time frame (column).

After applying both functions, I use the `matplotlib` library to visualize the spectrogram as an image. Below is the spectrogram of the first audio file:

![Spectrogram](plots/spec.png)

The spectrogram plot provides a visual representation of how the energy of different frequency bands varies over time.

Now that I have explored and processed the audio files into numerical values in the form of Mel spectrograms, I am ready to begin the modeling phase using CNNs.

---

# Convolutional Neural Networks

## Presentation

CNNs are deep learning models designed to process grid-like data, such as spectrograms. They use convolutional layers to extract features, pooling layers to reduce dimensionality, and fully connected layers for classification or regression. This architecture makes CNNs highly effective for tasks like image recognition, audio analysis, and more. I use the `pytoch` library to implement them in my notebook.

### Challenges
There are two main challenges to face:

- **Variable Shapes of Mel Spectrograms:** The number of columns $n$ in the spectrogram matrices varies due to differences in audio duration across files.
- **Small Dataset Size:** The dataset contains only 604 audio files, which may limit the model's ability to generalize effectively.

To address the variable shape issue, I will use two techniques. First, I will train CNNs with a batch size of 1 and an adaptive pooling layer, which can handle data of different sizes. Second, I will resize the spectrograms by applying padding and interpolation to standardize their dimensions.

To address the lack of data, I will attempt to expand the dataset using various data augmentation techniques.

### Activation and Loss Function

In all the CNNs I implemented, I used ReLU as the activation function and Cross-Entropy as the loss function.


#### ReLU (Rectified Linear Unit)

ReLU is an activation function defined as:

$$
f(x) = \max(0, x)
$$

ReLU is a computationally efficient activation function that outputs the input if positive or zero otherwise, introducing non-linearity to learn complex patterns, reducing the vanishing gradient problem, and performing well in tasks like image and audio classification.


#### Cross-Entropy

Cross-Entropy is a loss function commonly used for classification tasks. It measures the difference between the predicted probability distribution and the actual labels. The Cross-Entropy loss for a single sample is defined as:

$$
L = -\sum_{i=1}^C y_i \log(\hat{y}_i)
$$

Where:
- $C$ is the number of classes.
- $y_i$ is the ground truth label for class $i$ (1 if it is the correct class, 0 otherwise).
- $\hat{y}_i$ is the predicted probability for class $i$.


## CNNs with Batch Size of 1

Since my data has variable shapes, I first create a data loader with a **batch size of 1** and implement CNNs with **adaptive pooling**. Using a batch size of 1 allows the model to process one sample at a time, removing the requirement for all samples in a batch to have the same dimensions. 

Adaptive pooling is a specialized pooling layer in CNNs that dynamically adjusts the output size to a predefined value, regardless of the input dimensions. Unlike traditional pooling methods (e.g., max pooling or average pooling with fixed kernel sizes), adaptive pooling ensures consistency in the output feature map size, making it ideal for handling inputs with variable shapes.

I implemented two CNNs: one with 3 layers and another with 4 layers. I applied a 25% dropout to reduce overfitting by randomly deactivating neurons during training, improving the model's generalization ability.


### Results

#### ***3 Layers***

#### ***4 Layers***


## CNNs with Resized Data

To address the issue of variable shapes, another solution is to resize the spectrograms. The challenge here is to minimize the loss of information during resizing. To achieve this, I used two techniques: **padding** and **interpolation**.


### Padding

Padding involves adding extra values (usually zeros) to the edges of the spectrogram to make all spectrograms the same size. This technique preserves the original data entirely and ensures no loss of information. The zeros simply act as placeholders, and the original structure of the spectrogram remains intact. This is particularly useful when the differences in shape are small and padding can bridge the gap without affecting the overall data distribution.

### Interpolation

Interpolation resizes the spectrograms by scaling them up or down to a target size. This is done by estimating values for the missing or extra data points based on the original spectrogram. While interpolation can lead to slight alterations in the data, it allows the spectrograms to be standardized in size without introducing artificial structures. This technique is particularly effective when the differences in shape are significant and padding alone cannot resolve the issue.

I chose a target spectrogram dimension of $128 \times 256$, as the majority of spectrograms have more than 256 time frames. This ensures that minimal padding is required while retaining a sufficiently large dimension to preserve as much information as possible for spectrograms with the most time frames. Spectrograms with $n < 256$ will be padded, while those with $n > 256$ will be resized using interpolation.

ADD THE PLOT OF n REPARTION

After reshaping the data, I implemented two CNNs, one with 3 layers and the other with 4 layers, using 25% dropout, and max pooling for dimensionality reduction and feature extraction.

### Results

#### ***3 Layers***

#### ***4 Layers***

## CNNs on Artificial Data

### Method to Increase the Dataset

To address the small dataset size (604 audio files transformed into Mel spectrograms), I will artificially augment the data. By applying 4 augmentation methods randomly 10 times to each spectrogram, I aim to increase the dataset size by a factor of 10. The augmentation methods are as follows:

#### ***Add Random Noise***

Random noise is added to the spectrogram using a Gaussian distribution with a variance of $0.01$.


#### ***Pitch Shift***

The pitch of the audio is shifted by a random integer between $-4$ and $4$ of semitones within a specified range. This method alters the perceived pitch while preserving other characteristics, simulating variations in vocal or instrumental tones.


#### ***Frequency Mask***

A random frequency range in the spectrogram is masked (set to zero). This simulates scenarios where certain frequency bands are missing or obscured, helping the model become robust to incomplete data.


#### ***Time Mask***

A random time range in the spectrogram is masked. This technique mimics interruptions or short silences in the audio, forcing the model to learn from incomplete temporal patterns.


By combining these techniques, I aim to create a diverse and enriched dataset that enhances the model's ability to generalize and perform well on unseen data. Each technique will be applied with a $60\%$ probability, ensuring a balanced and varied augmentation process.

I trained my augmented dataset on two 4-layer CNNs: one using a batch size of 1, and the other using resized spectrograms.

### Results

#### ***Batch size of 1***

#### ***Resized spectrograms***

---

# Conclusion


