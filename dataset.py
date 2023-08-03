import nltk
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
    def __init__(self, text_data, labels, paragraph_size, review_size):
        self.combined = labels + text_data
        self.text_data = text_data
        self.labels = labels
        self.tokenizer = get_tokenizer('basic_english')
        self.data = []


        for text in self.combined:
            t = self.tokenizer(text)
            self.data.append(t)

        self.vocab = build_vocab_from_iterator(self.data, specials= ["<PAD>", "<UNK>", "<SOS>", "<EOS>"], min_freq=5)
        self.paragraph_size = paragraph_size
        self.review_size = review_size

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

        # Clean up the text
        text2 = self.clean_text(text)

        # Tokenize text and convert to numerical
        tokens = self.tokenizer(text2)

        # Create the numerical representation
        numerical_representation = []
        for token in tokens:
            if token not in self.vocab:
                numerical_representation.append(self.vocab["<UNK>"])
                continue
            numerical_representation.append(self.vocab[token])

        # Init SOS and EOS tokens
        SOS =  self.vocab["<SOS>"]
        EOS =  self.vocab["<EOS>"]

        # Add SOS token as the first value
        numerical_representation = [SOS] + numerical_representation

        # If numerical_representation is longer than self.paragraph_size, truncate it and add EOS token
        if len(numerical_representation) >= self.paragraph_size:
            numerical_representation = numerical_representation[:self.paragraph_size - 1]  # Make space for EOS
            numerical_representation.append(EOS)
        else:
            # Calculate the number of paddings required
            num_paddings = self.paragraph_size - len(numerical_representation) -1

            # Append EOS token if there is still space for it
            numerical_representation.append(EOS)

            # Pad the numerical_representation
            padding_array = np.array([self.vocab["<PAD>"]] * num_paddings)
            numerical_representation += padding_array.tolist()

        numerical_representation = torch.tensor(numerical_representation)

        ## Convert Label to tensor with numerical representation

        # Tokenize label and convert to numerical representation
        tokens_label = self.tokenizer(label)
        numerical_representation_label = []
        for token in tokens_label:
            if token not in self.vocab:
                numerical_representation_label.append(self.vocab["<UNK>"])
                continue
            numerical_representation_label.append(self.vocab[token])

        # Add SOS token as the first value in label
        numerical_representation_label = [SOS] + numerical_representation_label

        # If numerical_representation_label is longer than self.review_size, truncate it and add EOS token
        if len(numerical_representation_label) >= self.review_size:
            numerical_representation_label = numerical_representation_label[:self.review_size - 1]  # Make space for EOS
            numerical_representation_label.append(EOS)
        else:
            # Calculate the number of paddings required
            num_paddings_label = self.review_size - len(numerical_representation_label) - 1

            # Append EOS token if there is still space for it
            numerical_representation_label.append(EOS)

            # Pad the numerical_representation_label
            padding_array = np.array([self.vocab["<PAD>"]] * num_paddings_label)
            numerical_representation_label += padding_array.tolist()

        numerical_representation_label = torch.tensor(numerical_representation_label)

        return numerical_representation, numerical_representation_label