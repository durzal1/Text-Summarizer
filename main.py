import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from dataset import *
import nltk
from tqdm import tqdm


# Constants
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = .001
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DEVICE = "cpu"
embedding_dim = 100
hidden_dim = 100
NUM_LAYERS = 2
CLIP = 5

# Retrieve data
datapipe = IterableWrapper(["test.csv"])
datapipe = FileOpener(datapipe, mode='b')
datapipe = datapipe.parse_csv(skip_lines=1)

paragraph_data = []
summarized_data = []
cnt = 0
tokenizer = get_tokenizer('basic_english')
for sample in datapipe:
    cnt += 1

    # adds the data and separates it
    paragraph_data.append(sample[1])
    summarized_data.append(sample[2])

    # cnt here is used to ensure that it perfectly fits with my batch_size
    # if (cnt == 49984):
    #     break
    """
    Not sure if i need that anymore I will see later on
    """

# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(paragraph_data, summarized_data, test_size=0.2, random_state=1)


# Combine the X and Y and adjust it before converting to a tensor
train_data = CustomDataset(x_train, y_train, 500, 50)
test_data = CustomDataset(x_test, y_test, 500, 50)

train_data.__getitem__(0)

# create tensor dataset
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

# Important variables (Add 2 because of SOS and EOS tokens)
input_vocab_size = len(train_data.vocab) + 3
output_vocab_size = len(train_data.vocab) + 3



class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

# Step 4: Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.output_vocab_size = output_vocab_size
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
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_vocab_size, hidden_size, num_layers=num_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, num_layers=num_layers)
    """
    NOTE: Every tensor in the batch is being computed at the same time
    """
    def forward(self, input_seq, target_seq):

        # Shape should be (paragraph_size, BATCH_SIZE)
        input_seq = input_seq.transpose(0, 1)
        target_seq = target_seq.transpose(0, 1)

        # Initialize the hidden layer
        h0 = torch.zeros((self.decoder.num_layers, BATCH_SIZE, self.hidden_size)).to(DEVICE)
        c0 = torch.zeros((self.decoder.num_layers, BATCH_SIZE, self.hidden_size)).to(DEVICE)
        hidden = (h0, c0)

        encoder_output, encoder_hidden = self.encoder(input_seq, hidden)

        # Get the maximum sequence length for the target sequence
        max_len = target_seq.size(0)

        # Initialize the output tensor to store the generated sequence
        decoder_outputs = torch.zeros(max_len, BATCH_SIZE, self.decoder.output_vocab_size).to(input_seq.device)

        # Start the decoding process with the start-of-sequence token as the first input
        decoder_input = target_seq[0, :].unsqueeze(0)       # Use the start-of-sequence token from the target sequence


        # Iterate over the target sequence and generate the output
        for t in range(1, max_len):
            decoder_output, encoder_hidden = self.decoder(decoder_input, encoder_hidden)
            decoder_outputs[t] = decoder_output

            # Use teacher forcing: set the next decoder input to be the true target token at the current time step
            decoder_input = target_seq[t, :].unsqueeze(0)

        return decoder_outputs


# Training

# Initialize the model, loss function, and optimizer
model = Seq2Seq(input_vocab_size, output_vocab_size, hidden_dim, num_layers=NUM_LAYERS)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Set the model to training mode
model.train()

# Training loop
for epoch in range(NUM_EPOCHS):
    total_loss = 0

    loop = tqdm(train_loader, leave=True)

    for batch, (x,y) in enumerate(loop):
        # Transfer the data to the device (GPU if available)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Forward pass
        model.zero_grad()

        output = model(x,y)

        # Reshape the output and target for computing loss
        output = output.view(-1, output_vocab_size)
        target = y.view(-1)

        # Compute the loss
        loss = criterion(output, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # help prevent exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        # update gradients
        optimizer.step()

        total_loss += loss.item()

    # Calculate the average loss for the epoch
    average_loss = total_loss / len(train_loader)

    # Print the progress
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}")

# Save the trained model if desired
# torch.save(model.state_dict(), "text_summarizer_model.pth")