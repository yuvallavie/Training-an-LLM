"""Training a sentiment classifier

This project trains a simple RNN + MLP to classify a binary sentiment
analysis task, based on the amazon review dataset at https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data
We create a vocabulary from the training set and train a static embedding function from scratch.

"""
# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from training_utils import train, save_dictionary
from dataset import GetData
from text_processing import create_vocab, generate_dictionaries
from model import Classifier
from pathlib import Path
# %% Setting the paths for the training/dev/ files.
# Set the file paths for the training, dev datasets.
training_file_path = Path(r"dataset\split\train.txt")
dev_file_path = Path(r"dataset\split\dev.txt")
# %% Creating a vocabulary and dictionaries.
# Create a vocabulary from the training set
# which is usually the biggest set.
print("Creating the vocabulary")
vocab = create_vocab(training_file_path)

# Create both dictionaries which help us
# attach an embedding to a token later.
print("Creating the dictionaries (w2i, i2w)")
word_to_index, index_to_word = generate_dictionaries(vocab)
# %%
# Save the dictionary for later use.
save_dictionary(word_to_index, "word_to_index")
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

# Loss function for classification (CE)
criterion = nn.CrossEntropyLoss()

# Optimizer & Learning rate.
# We use a learning rate of 1e-03.
# We can also use a scheduler, but this task
# finishes in 4-5 epochs and its not needed.
learning_rate = 0.001
# Adam optimizer.
optimizer = optim.Adam(clf.parameters(), lr=learning_rate)

# Device configuration (CPU or GPU)
device = torch.device('cuda')
# Move the classifier to the GPU.
clf.to(device)
# %% Load the data.
# We load the data using our customer PyTorch
# Dataloaders. The batch size has a big effect
# on the success of the training. While the GPU
# Can handle huge batch sizes, it doesnt bode well
# for the training. We choose a batch size of 12.
# Create the data loaders.
batch_size = 12
# Loading the training set.
training_loader = GetData(training_file_path,
                          batch_size=batch_size, w2i=word_to_index)
# Loading the validation set.
validation_loader = GetData(dev_file_path,
                            batch_size=batch_size, w2i=word_to_index)

# Train and validation data loaders.
# Our training function requires a dictionary
# of dataloaders which have "train" and "validate" keys.
loaders = {
    "train": training_loader,
    "validate": validation_loader
}

# %% Train the classifier
# Number of training epochs
num_epochs = 5
# Train the classifier.
results = train(clf, optimizer, loaders, criterion, num_epochs, device)
