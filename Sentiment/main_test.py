# %% Imports
import torch
from model import Classifier
from pathlib import Path
from training_utils import load_dictionary, predict, display_classification_results
from dataset import GetData

# %% Loading necessary files.
# Set the path to the test file.
test_file_path = Path(r"dataset\split\test.txt")
# Load the word-to-index dictionary.
word_to_index = load_dictionary("word_to_index.json")
# Load the best model weights.
model_weights = torch.load("best_model_weights.pth")

# %% Model & Optimizer.
# The dimension of the embedding layer.
embedding_dim = 200
# The outout dimension of the RNN.
hidden_dim = 100
# The number of classes.
num_classes = 2
# Initialize the classifier.
# The classifier creates an embedding matrix the size of
# len(word_to_index) and assumes that its inputs are vectors
# or matrices of indices.
clf = Classifier(len(word_to_index), embedding_dim, hidden_dim, num_classes)
# %% Load the model parameters and the test data.
clf.load_state_dict(model_weights)
# Load the model to the GPU.
device = torch.device("cuda")
clf.to(device)
# Load the test data, we can
# use a greater batch size because
# we wont be propogating any gradients.
test_loader = GetData(test_file_path,
                      batch_size=1028, w2i=word_to_index)

# %% Predict on the test set.
print("-" * 30)
print("  Predicting on the test set")
print("-" * 30)
# Predict using the trained model.
y_pred, y_true = predict(clf, test_loader)

# %% Display the results of classification.
# Print the classification result.
print("-" * 23)
print("Classification results")
print("-" * 23)
display_classification_results(y_true, y_pred)
