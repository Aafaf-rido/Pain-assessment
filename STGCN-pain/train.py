import numpy as np
random_seed = 42  # for reproducibility
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
# from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from GCN.data_processing import Data_Loader
from GCN.graph import Graph
from GCN.sgcn_lstm import Sgcn_Lstm
from sklearn.metrics import  f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import argparse


# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument('--data', type=str, default='landmarkdata',
                       help='the name of the data dir')

my_parser.add_argument('--lr', type=int, default= 0.0001,
                       help='initial learning rate for optimizer.')

my_parser.add_argument('--epoch', type=int, default= 1000,
                       help='number of epochs to train.')

my_parser.add_argument('--batch_size', type=int, default= 10,
                       help='training batch size.')
#my_parser.add_argument('Path',
#                       type=str,
#                       help='the path to list')

# Execute the parse_args() method
args = my_parser.parse_args()

# C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.1.3\host-windows-x64
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin
 

"""import the whole dataset"""
data_loader = Data_Loader(args.data)  # folder name -> Train.csv, Test.csv

"""import the graph data structure"""
graph = Graph(data_loader.num_landmarks)

"""Split the data into training and validation sets while preserving the distribution"""
train_x, test_x, train_y, test_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state = random_seed)


"""Train the algorithm"""
algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr = args.lr, epoach=args.epoch, batch_size=args.batch_size)
history = algorithm.train()







# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

plot_history(history)

"""Test the model"""

y_pred = algorithm.prediction(test_x)

# Convert probabilities to class labels (assuming binary classification with softmax output)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ensure that test_y is in the correct format
test_y_flat = test_y.flatten()

# Ensure both arrays are integer type
y_pred_classes = y_pred_classes.astype(int)
test_y_flat = test_y_flat.astype(int)

# Check the shapes and types
print(f"Shape of y_pred_classes: {y_pred_classes.shape}, dtype: {y_pred_classes.dtype}")
print(f"Shape of test_y_flat: {test_y_flat.shape}, dtype: {test_y_flat.dtype}")

# Compute accuracy
accuracy = np.mean(y_pred_classes == test_y_flat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Compute F1 score, precision, and recall
f1 = f1_score(test_y_flat, y_pred_classes, average='weighted')
precision = precision_score(test_y_flat, y_pred_classes, average='weighted')
recall = recall_score(test_y_flat, y_pred_classes, average='weighted')

print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


conf_matrix = confusion_matrix(test_y_flat, y_pred_classes)

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Define class names (change these based on your actual class labels)
class_names = ['no pain', 'pain']

plot_confusion_matrix(conf_matrix, class_names)




