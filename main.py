import random

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
from nltk.translate.bleu_score import corpus_bleu


# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = .04
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
# DEVICE = "cpu"
embedding_dim = 64
hidden_dim = 64
NUM_LAYERS = 2
CLIP = 5
DROPOUT = .5

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
x_train, x_test, y_train, y_test = train_test_split(paragraph_data, summarized_data, test_size= .1, random_state=1)


# Combine the X and Y and adjust it before converting to a tensor
train_data = CustomDataset(x_train, y_train, 500, 50)
test_data = CustomDataset(x_test, y_test, 500, 50)

train_data.__getitem__(0)

# create tensor dataset
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

# Important variables (Add 3 because of SOS and EOS tokens)
input_vocab_size = len(train_data.vocab) + 3
output_vocab_size = len(train_data.vocab) + 3


# Experimental
class ConcatenateAttention(nn.Module):
    def __init__(self, hidden_dimmension):
        super(ConcatenateAttention, self).__init__()

        self.hidden_dimmension = hidden_dimmension

        self.linear_in = nn.Linear(hidden_dimmension + hidden_dimmension, hidden_dimmension)
        self.v = nn.Linear(hidden_dimmension, 1, bias=False)


    def forward(self, decoder_hidden, encoder_outputs):

        seq_length, batch_size, _ = encoder_outputs.size()

        decoder_hidden = decoder_hidden[-1]  # Take the last layer's hidden state
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_length, 1)
        decoder_hidden = decoder_hidden.permute(1,0,2)
        combined = torch.cat((encoder_outputs, decoder_hidden), dim=2)

        energy = torch.tanh(self.linear_in(combined))

        attention_scores = torch.softmax(self.v(energy), dim=2)

        # Permute attention_scores so that batch_size is the first dimension
        attention_scores = attention_scores.permute(1, 0, 2)

        attention_scores = attention_scores.permute(0, 2, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Compute weighted context by matrix multiplication
        weighted_context = torch.bmm(attention_scores, encoder_outputs)

        # Permute the result back to the original shape [batch_size, 1, hidden_dimmension]
        weighted_context = weighted_context.permute(1, 0, 2)

        # Squeeze to remove the middle dimension
        weighted_context = weighted_context.squeeze(1)

        return weighted_context, attention_scores

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, input_seq, hidden, padding_mask):
        # input_seq = input_seq.transpose(0,1)
        embedded = self.dropout(self.embedding(input_seq))

        # Calculate input_lengths from the padding_mask
        input_lengths = padding_mask.sum(dim=1)

        # Pack the padded input sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.to("cpu"), enforce_sorted=False)

        output, hidden = self.rnn(packed_embedded , hidden)

        # Unpack the packed sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        return output, hidden

# Step 4: Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()

        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, output_vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

        # Attention mechanism
        self.attention = ConcatenateAttention(hidden_size)

    def forward(self, input_seq, hidden, encoder_output):

        # input_seq = input_seq.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_seq))

        context, _ = self.attention(hidden[0], encoder_output)

        concat_embedded = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(concat_embedded, hidden)

        output = self.output_layer(output)

        return output, hidden


# Step 5: Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size,num_layers=1):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_vocab_size, hidden_size, num_layers = num_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, num_layers = num_layers)
    """
    NOTE: Every tensor in the batch is being computed at the same time
    """
    def forward(self, input_seq, target_seq, padding_mask, teacher_forcing_ratio):
        teacher_forcing_ratio = 1
        # Shape should be (paragraph_size, BATCH_SIZE)

        batch = input_seq.shape[1]

        # Initialize the hidden layer
        h0 = torch.zeros((self.decoder.num_layers, batch, self.hidden_size)).to(DEVICE)
        c0 = torch.zeros((self.decoder.num_layers, batch, self.hidden_size)).to(DEVICE)
        hidden = (h0, c0)

        encoder_output, encoder_hidden = self.encoder(input_seq, hidden, padding_mask)

        # Get the maximum sequence length for the target sequence
        max_len = target_seq.size(0)

        # Initialize the output tensor to store the generated sequence
        decoder_outputs = torch.zeros(max_len, batch, self.decoder.output_vocab_size).to(input_seq.device)

        # Start the decoding process with the start-of-sequence token as the first input
        decoder_input = target_seq[0, :].unsqueeze(0)       # Use the start-of-sequence token from the target sequence


        # Iterate over the target sequence and generate the output
        for t in range(1, max_len):
            decoder_output, encoder_hidden = self.decoder(decoder_input, encoder_hidden, encoder_output)
            decoder_outputs[t] = decoder_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = decoder_output.argmax(2)

            # Use teacher forcing: set the next decoder input to be the true target token at the current time step
            decoder_input = target_seq[t, :].unsqueeze(0) if teacher_force else top1

            # print('f')

        return decoder_outputs


def tensor_to_text(g):
    ids = g.squeeze().tolist()

    id_to_word = {idx: word for idx, word in enumerate(train_data.vocab.get_itos())}

    words = []

    if g.dim() == 1 or g.ndim==1:
        values = g
    else:
        # Get the IDS of the words
        values = g.argmax(1)

     # Convert integer IDs to words using the id_to_word dictionary
    for word in values:
        words.append(id_to_word[word.item()])


    # Join the words to form the text summary
    text_summary = " ".join(words)
    return text_summary

# Training

# Initialize the model, loss function, and optimizer
model = Seq2Seq(input_vocab_size, output_vocab_size, hidden_dim, num_layers=NUM_LAYERS)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=test_data.vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Set the model to training mode
model.train()

# Test
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


# model.apply(init_weights)

# model.load_state_dict(torch.load("check2.pth"))

# Just some Info to know how long backprop will take
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# Training loop


for epoch in range(NUM_EPOCHS):
    total_loss = 0

    loop = tqdm(train_loader, leave=True)

    generated_summaries = []
    reference_summaries = []
    cnt = 0
    for batch, (x,y) in enumerate(loop):

        # Transfer the data to the device (GPU if available)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        padding_mask = (x != train_data.vocab["<PAD>"]).float()

        x = x.transpose(0, 1)
        y = y.transpose(0, 1)

        # Create a padding mask for the input sequence

        optimizer.zero_grad()

        output = model(x,y, padding_mask, 0)

        # Reshape the output and target for computing loss
        output2 = output.view(-1, output_vocab_size)
        target2 = y.reshape(-1)

        # Compute the loss
        loss = criterion(output2, target2)

        # Backpropagation
        loss.backward()

        # help prevent exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        # update gradients
        optimizer.step()

        total_loss += loss.item()

        # Check accuracy
        output = output.transpose(0,1)
        y = y.transpose(0,1)

        # if (cnt == 161):
        #
    #     generated_summaries.extend([tensor_to_text(g) for g in output])
    #     reference_summaries.extend([tensor_to_text(t) for t in y])
    #     print(generated_summaries)
    #     print(reference_summaries)
    #     cnt += 1
    #     break
    # break

    # Calculate the average loss for the epoch
    average_loss = total_loss / len(train_loader)
    # bleu_score = corpus_bleu(reference_summaries, generated_summaries)

    # Print the progress
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}")
    # print("BLEU Score:", bleu_score)
# Save the trained model if desired
torch.save(model.state_dict(), "check1.pth")