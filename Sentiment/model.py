"""Classifier for the project.

This file holds the torch model used for binary classification.
"""
import torch
import torch.nn as nn
from torch.nn import Embedding, LSTM, Linear
from torch.nn.init import xavier_uniform_, orthogonal_, kaiming_uniform_


class Classifier(torch.nn.Module):
    """Binary classifier for the task.


    This class holds the binary classifier, we train an embedding layer
    and a single LSTM with a fully connected layer.
    This model should be trained with the CrossEntropyLoss.
    """

    def __init__(self, dict_size, embedding_dim, hidden_dim, num_classes) -> None:
        super(Classifier, self).__init__()
        # Initialize the embedding layer.
        self.embedding_layer = Embedding(dict_size, embedding_dim)
        # Initialize the LSTM.
        self.rnn = LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Initialize the MLP.
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), num_classes)
        )

        # Initialize LSTM
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # Turn the sentence into indices.
        x = self.embedding_layer(x)
        # Run it through the LSTM.
        _, (h, _) = self.rnn(x)
        # Run it through the MLP.
        x = self.fc(h)
        # Return the distribution.
        return x

    def predict(self, inputs=None, logits=None):
        """Predict on inputs or logits
        """
        if (logits == None):
            # Forward pass.
            logits = self.forward(inputs)
        # Predict.
        return torch.argmax(logits.squeeze(), dim=1)
