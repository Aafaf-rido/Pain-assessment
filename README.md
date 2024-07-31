# Pain-assessment
Assessment of patient pain using deep learning approaches with facial expressions

## General description 

This repository contains two neural network models for pain classification using the Pain Emotion Faces Database (PEMF).

1. __ConvNeXt-LSTM Model__: This model extracts spatial features from video frames in the dataset using a pretrained ConvNeXt XLarge network. The extracted features are then passed to an LSTM model to capture the dependencies between frames in a video.


![CNN-LSTMglobal](https://github.com/user-attachments/assets/4accacf6-742f-4f69-a472-6528123b7291)


2. __STGCN-LSTM Model__: This model uses the facial landmarks from video frames to address the temporal aspect of the data. The landmarks are treated as input to an STGCN-LSTM model, which processes these landmarks as a graph based on the connections between the 68 facial landmarks.

![STGCN](https://github.com/user-attachments/assets/8f6af47f-159c-4daf-8836-ffa5e6669c2f)


Both models have a classification layer at the end for binary classification (pain/no pain).

The PEMF dataset comprises 272 micro-clips, each represented by 20 frames. We also extracted the landmarks of these frames to serve as input to the STGCN-LSTM model. The landmarks are organized into a graph structure based on the relationships between the 68 facial landmarks.

## Quick Start

1. Load the PEMF dataset and change the path variable in the file hybridCovNeXt.py
   
2. create and activate a conda environment

`conda create --name pain anaconda`

`conda activate pain`

3. Install the requirements

`pip install -r requirements.txt`

4. Run the code

for the Hybrid ConvNeXt:

`python hybridConvNeXt.py`

for the STGCN Model

`python train.py`
