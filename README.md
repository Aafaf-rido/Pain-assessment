# Pain-assessment
Assessment of patient pain using deep learning approaches with facial expressions

## General description 

This repository contains two neural network models for pain classification using the Pain Emotion Faces Database (PEMF).

1. ConvNeXt-LSTM Model: This model extracts spatial features from video frames in the dataset using a pretrained ConvNeXt XLarge network. The extracted features are then passed to an LSTM model to capture the dependencies between frames in a video.

2. STGCN-LSTM Model: This model uses the facial landmarks from video frames to address the temporal aspect of the data. The landmarks are treated as input to an STGCN-LSTM model, which processes these landmarks as a graph based on the connections between the 68 facial landmarks.

Both models have a classification layer at the end for binary classification (pain/no pain).

The PEMF dataset comprises 272 micro-clips, each represented by 20 frames. We also extracted the landmarks of these frames to serve as input to the STGCN-LSTM model. The landmarks are organized into a graph structure based on the relationships between the 68 facial landmarks.
