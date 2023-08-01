import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from dataset import *

# Constants
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = .001
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
embedding_dim = 100
hidden_dim = 100
NUM_LAYERS = 2
CLIP = 5

# Retrieve data
datapipe = IterableWrapper(["IMDB Dataset.csv"])
datapipe = FileOpener(datapipe, mode='b')
datapipe = datapipe.parse_csv(skip_lines=1)

paragraph_data = []
summarized_data = []
cnt = 0
tokenizer = get_tokenizer('basic_english')
for sample in datapipe:
    cnt += 1

    # adds the data and separates it
    paragraph_data.append(sample[0])
    summarized_data.append(sample[1])

    # cnt here is used to ensure that it perfectly fits with my batch_size
    # if (cnt == 49984):
    #     break
    """
    Not sure if i need that anymore I will see later on
    """

# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(paragraph_data, summarized_data, test_size=0.2, random_state=1)


# Combine the X and Y and adjust it before converting to a tensor
train_data = CustomDataset(x_train, y_train,)
test_data = CustomDataset(x_test, y_test)


# create tensor dataset
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded)
        return output, hidden

# Step 4: Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        output = self.output_layer(output)

        return output, hidden


# Step 5: Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, num_layers=1):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_vocab_size, hidden_size, num_layers=num_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, num_layers=num_layers)

    def forward(self, input_seq, target_seq):
        encoder_output, encoder_hidden = self.encoder(input_seq)

        # Initialize the decoder's hidden state with the encoder's final hidden state
        decoder_hidden = encoder_hidden

        # Get the maximum sequence length for the target sequence
        max_len = target_seq.size(0)

        batch_size = target_seq.size(1)  # Get the batch size

        # Initialize the output tensor to store the generated sequence
        decoder_outputs = torch.zeros(max_len, batch_size, self.decoder.output_vocab_size).to(target_seq.device)

        # Start the decoding process with the start-of-sequence token as the first input
        decoder_input = target_seq[0, :]  # Use the start-of-sequence token from the target sequence

        # Iterate over the target sequence and generate the output
        for t in range(1, max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_output

            # Use teacher forcing: set the next decoder input to be the true target token at the current time step
            decoder_input = target_seq[t, :]

        return decoder_outputs


# Training
