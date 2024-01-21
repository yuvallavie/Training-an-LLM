from torch import tensor
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def generate_dictionaries(unique_tokens: set) -> dict:
    """Generate Word-To-Index and Index-To-Word.

    This function receives a set of unique tokens and
    returns two dictionarys.
    1. Word-To-Index.
    2. Index-To-Word.
    """
    # Word to index and Index to word dictionaries.
    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1}
    index_to_word = {
        0: "<PAD>",
        1: "<UNK>"}

    # Starting index.
    index = 2
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
    tokens = preprocess_and_tokenize(sentence)
    return tensor([w2i[token] if token in w2i else w2i["<UNK>"] for token in tokens])


def preprocess_and_tokenize(sentence: str) -> list[str]:
    """Proprocess a sentence and tokenize.

    This function takes a sentence and preprocesses it in a
    way that helps models understand. It uses Lemmatization,
    lowering and opening contractions. It eventually
    returns a list of tokens
    """
    # Expand contradictions, don't -> do not.
    sentence = contractions.fix(sentence)
    # Transform the sentence to lowercase.
    sentence = sentence.lower()
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    # Lemmatize the tokens.
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Return the resulting tokens.
    return tokens


def create_vocab(filePath: str) -> set:
    """Create a vocab using lazy loading.

    This function creates a unique vocabulary but does not 
    load the entire dataset into the memory. Datasets can be huge
    and are not meant to be loaded directly to the RAM.
    It expects the file to have entries with the following format:
    __label__y TEXT
    An example:
    __label__negative This movie is not so good!
    __label__positive I really liked this movie.
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
            tokens = preprocess_and_tokenize(sentence)
            # Insert the tokens to a unique set.
            unique_tokens.update(tokens)
    # Return the unique tokens.
    return unique_tokens
