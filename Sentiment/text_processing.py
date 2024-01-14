from nltk.tokenize import word_tokenize
import torch


def create_vocab(sentences: list) -> set:
    """Creating a unique vocabulary from a list of sentences

    This function expects to receive a list of sentences in english
    and returns a unique set of tokens from those sentences.
    """
    # A unique empty set.
    unique_tokens = set()

    # Run over all sentences in the list.
    for sentence in sentences:
        # Lower case the sentence.
        sentence = sentence
        # Tokenize the sentence.
        tokens = word_tokenize(sentence)
        # Insert each token to a unique set
        # which keeps consistency.
        for token in tokens:
            unique_tokens.add(token)

    # Return the vocabulary.
    return unique_tokens


def generate_dictionaries(unique_tokens: set) -> dict:
    """Generate Word-To-Index and Index-To-Word.

    This function receives a set of unique tokens and
    returns two dictionarys.
    1. Word-To-Index.
    2. Index-To-Word.
    """
    # Word to index and Index to word dictionaries.
    word_to_index = {"<UNK>": 0}
    index_to_word = {0: "<UNK>"}
    # Starting index.
    index = 1
    # Run over all tokens and add them accordingly.
    for token in unique_tokens:
        # Add the token to both dictionaries.
        word_to_index[token] = index
        index_to_word[index] = token
        # Increase the index.
        index += 1

    # Return both dictionaries.
    return word_to_index, index_to_word

# Function to convert sentences to index tensor


def sentence_to_indices(sentence: str, w2i: dict):
    """Create a tensor of indices from a sentence

    This function separates a sentence into tokens
    and returns a tensor of indices for each.
    """
    tokens = word_tokenize(sentence)
    return torch.tensor([w2i[token] if token in w2i else w2i["<UNK>"] for token in tokens])


def create_vocab_lazy(filePath: str) -> set:
    """Create a vocab using lazy loading.

    This function creates a unique vocabulary but does not 
    load the entire dataset into the memory. Datasets can be huge
    and are not meant to be loaded directly to the RAM.
    It expects the file to have entries with the following format:
    __label__y TEXT
    An example:
    __label__1 This movie is not so good!
    """
    # Empty token set.
    unique_tokens = set()
    # Open the data set text file.
    with open(filePath, 'r', encoding='utf-8') as file:
        # Read line by line.
        for line in file:
            # Separate the line to (label, text).
            label, sentence = line.split(' ', 1)
            # Tokenize the text.
            tokens = word_tokenize(sentence.lower())
            # Insert the tokens to a unique set.
            for token in tokens:
                unique_tokens.add(token)
    # Return the unique tokens.
    return unique_tokens
