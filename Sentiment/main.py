"""Training a sentiment classifier


based on the amazon review dataset at https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data
"""

# %%
"""Loading the dataset.

This part loads a dataset or a dataloader which can be used
to learn with a classifier.
"""

# Create the dictionaries for text processing.
from training_utils import train
from model import Classifier
import torch.optim as optim
import torch.nn as nn
import torch
from dataset import GetData
from text_processing import create_vocab_lazy, generate_dictionaries
print("Creating the vocabulary")
vocab = create_vocab_lazy("small_train.txt")
print("Creating the dictionaries (w2i, i2w)")
w2i, i2w = generate_dictionaries(vocab)
# %%
# Create the data loaders.
batch_size = 4096
training_loader = GetData("small_train.txt", batch_size=batch_size, w2i=w2i)
validation_loader = GetData("small_dev.txt", batch_size=batch_size, w2i=w2i)
# %%

# Initialize the classifier.
clf = Classifier(len(vocab), 300, 100, w2i)

# Loss function for binary classification (BCE)
criterion = nn.CrossEntropyLoss()

# Optimizer (e.g., Adam)
optimizer = optim.Adam(clf.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 10

# Device configuration (CPU or GPU)
device = torch.device('cuda')
clf.to(device)

# Train and validation data loaders.
loaders = {
    "train": training_loader,
    "validate": validation_loader
}

# %%
train(clf, optimizer, loaders, criterion, num_epochs, device)
