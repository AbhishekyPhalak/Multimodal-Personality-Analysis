# Multimodal Personality Analysis for Candidate Screening

## Overview

This project leverages deep learning techniques to extract meaningful information from both visual and audio modalities of video clips in order to assess an individual's personality. The goal is to provide valuable insights into candidates' behavioral traits, which can be used for more informed decision-making during the candidate screening process.

### Key Features:
- Personality trait prediction from video data (Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness)
- Combination of visual and audio modalities for enhanced accuracy
- Deep learning-based feature extraction and model training
- Use of state-of-the-art models like LSTM, CNN, and VGG16 for temporal and facial feature processing

## Problem Definition

Understanding human personality plays an essential role in evaluating behavioral traits for professional and personal development. In this project, we address the challenge of personality prediction from short video sequences, integrating both visual and audio features to improve accuracy.

This project’s primary aim is to develop a solution that assists in candidate screening by providing insights into personality traits through video content, improving recruitment decisions, reducing turnover, and enhancing employee satisfaction.

## Literature Review

In recent years, deep learning techniques have been applied to personality analysis, focusing on extracting insights from video data. Studies such as those by Subramaniam et al. [8], Wei et al. [2], and Kaya et al. [4] have successfully demonstrated the utility of multimodal learning for predicting personality traits based on visual and audio features.

## Dataset

The dataset used for training and evaluation in this project is the **Chalearn First Impressions V2** dataset, which consists of over 10,000 clips from more than 3,000 distinct YouTube videos. These clips contain diverse individuals speaking directly to the camera, allowing for a wide representation of gender, age, and ethnicity. The dataset is labeled with personality traits from the Five Factor Model: Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness.

## Methodology

### Preprocessing

#### Audio Preprocessing:
- Extract audio from video using ffmpeg
- Compute Mel-Frequency Cepstral Coefficients (MFCC) for each audio sample
- Standardize and reshape the MFCCs to prepare for model input

#### Video Preprocessing:
- Extract and resize frames from videos
- Perform face detection and cropping to focus on relevant facial features
- Stack processed frames into a 4D array (samples, height, width, channels)

#### Advanced Preprocessing:
- Audio: Use YAMNet to extract embeddings from raw audio data
- Video: Use the pre-trained VGGFace model to extract facial features

### Model Architecture

#### Audio Model:
- CNN-based architecture for feature extraction from the audio signal
- LSTM layers to capture the temporal dynamics of the audio data

#### Video Model:
- Use VGG16 pre-trained on facial data to extract features from video frames
- Feed the extracted features into an LSTM layer to capture temporal dependencies across video frames

### Fusion:
- The final output combines the features from both the audio and visual models
- Both modalities are fused to predict the personality traits using a regression-based model

## Results

We performed experiments on the Chalearn First Impressions V2 dataset, where the multimodal approach significantly outperformed unimodal models in predicting personality traits. The combined audio-visual features led to improved accuracy in assessing traits such as Extraversion and Openness, where visual features played a more significant role, while audio features were more crucial for traits like Neuroticism.

## Installation

To run the project on your machine, follow the instructions below:

### Requirements:
- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- librosa
- ffmpeg
- numpy
- pandas
- scikit-learn

### Setup:

Clone this repository:
   ```bash
   git clone https://github.com/your-repository/multimodal-personality-analysis.git
   cd multimodal-personality-analysis
