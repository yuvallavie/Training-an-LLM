"""Classifier for the project.

This file holds the torch model used for binary classification.
"""
import torch
from torch.nn import Embedding, LSTM, Linear

class Classifier(torch.nn.Module):
    """Binary classifier for the task.


    This class holds the binary classifier, we train an embedding layer
    and a single LSTM with a fully connected layer.
    This model should be trained with the CrossEntropyLoss.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, w2i ) -> None:
        super(Classifier,self).__init__()
        # Set the Word-To-Index dictionary.
        self.w2i = w2i
        # Initialize the embedding layer.
        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim)
        # Initialize the LSTM.
        self.rnn = LSTM(embedding_dim, hidden_dim, batch_first = True)
        # Initialize the MLP.
        self.fc = Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Turn the sentence into indices.
        x = self.embedding_layer(x)
        # Run it through the LSTM.
        _, (h,_) = self.rnn(x)
        # Run it through the MLP.
        x = self.fc(h)
        # Return the distribution.
        return x