import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import re
import html
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))

class CustomDataset(Dataset):
    def __init__(self, text_data, labels, paragraph_size):
        self.text_data = text_data
        self.labels = labels
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self.tokenizer(text) for text in text_data)
        self.paragraph_size = paragraph_size

    def __len__(self):
        return len(self.text_data)

    def clean_text(self, text):

        ## Convert text to lowercase
        text = text.lower()

        # Remove HTML tags from text using regular expression
        text = re.sub(r'<.*?>', '', text)

        # Remove punctuation
        text = ''.join([c for c in text if c not in string.punctuation])

        # Tokenize the text
        tokens = self.tokenizer(text)

        # Remove stop words (if you have a list of stop words named stop_words)
        tokens = [word for word in tokens if word not in stop_words]

        # Convert the list of tokens back to a string
        cleaned_text = ' '.join(tokens)

        return cleaned_text


    def __getitem__(self, idx):

        text = self.text_data[idx]
        label = self.labels[idx]

        if label == "positive":
            label = 1
        else:
            label = 0

        # clean up the text
        text = self.clean_text(text)

        features = np.zeros(self.reviews_size, dtype=int)

        # tokenize text and convert to numerical
        tokens = self.tokenizer(text)

        # create the numerical representation
        numerical_representation = []

        for token in tokens:
            if token not in self.vocab:
                continue
            numerical_representation.append(self.vocab[token])

        numerical_representation = torch.tensor(numerical_representation)

        # updates features with the values of numerical representations
        # adds padding if there is extra room and cuts if words is over self.reviews_size
        features[:len(numerical_representation)] = numerical_representation[:self.reviews_size]

        return features, label
