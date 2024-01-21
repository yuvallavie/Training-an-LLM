"""Creating a Torch dataset.


This file creates a Torch dataset from the Amazon data. It uses
lazy loading as we will not load the entire data to the memory when training.
"""
# %%
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from text_processing import sentence_to_indices


class ImdbDataset(Dataset):
    def __init__(self, filePath, w2i):
        """Initialize the database

        This function initializes the dataset. It sets
        the filePath and the word-to-index dictionary.
        """
        print("Initializing the dataset:", filePath)
        self.filePath = filePath
        self.w2i = w2i

    def __len__(self):
        """Count the lines of the file"""
        with open(self.filePath, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)

    def __getitem__(self, idx):
        """Get an item from the dataset

        This function reads a line from the text file
        which is expected to be in the FastText format
        and creates an item.
        """
        with open(self.filePath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == idx:
                    label, sentence = line.strip().split(' ', 1)
                    label = label = 1 if label == '__label__positive' else 0
                    indices = sentence_to_indices(sentence, self.w2i)
                    return indices, label


# Custom collate function for padding
def collate_fn(batch):
    """Pad each example in the batch"""
    sequences, labels = zip(*batch)
    # Pad the sequences to have the same length
    # pad_sequence automatically converts list of sequences into a padded tensor
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0)
    # Convert labels to a tensor
    labels = tensor(labels)
    return padded_sequences, labels


def GetData(filePath, batch_size, w2i):
    """Get a dataloader which works with padded batches.

    This function loads the dataset and then creates a dataloader
    that can be used with padded batches. It also returns the W2I, I2W
    and the size of the vocab.
    """
    # Create the dataset.
    dataset = ImdbDataset(filePath=filePath, w2i=w2i)

    # Return the data.
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
