import torch
import torch.nn as nn


def collate_fn(batch):
    """
    Create a batch of data given a list of N sequences and labels. Sequences are stacked into a single tensor
    of shape (N, max_sequence_length), where max_sequence_length is the maximum length of any sequence in the
    batch. Sequences shorter than this length should be filled up with 0's. Also returns a tensor of shape (N, 1)
    containing the label of each sequence.

    :param batch: A list of size N, where each element is a tuple containing a sequence tensor and a single item
    tensor containing the true label of the sequence.

    :return: A tuple containing two tensors. The first tensor has shape (N, max_sequence_length) and contains all
    sequences. Sequences shorter than max_sequence_length are padded with 0s at the end. The second tensor
    has shape (N, 1) and contains all labels.
    """

    sentences, labels = zip(*batch)

    N = len(sentences)
    max_sequence_length = max([len(s) for s in sentences])
    PaddedBatch = torch.zeros((N, max_sequence_length), dtype=torch.int32)
    Labels = torch.zeros((N, 1), dtype=torch.int32)
    for s in range(N):
        n = len(sentences[s])
        PaddedBatch[s, 0:n] = sentences[s]
        Labels[s, 0] = labels[s]

    return (PaddedBatch, Labels)

class RNNBinaryClassificationModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix

        # Variables
        input_size = embedding_dim
        hidden_size = 64

        # RNN
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # GRU
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.
        :param inputs: Tensor of shape (N, max_sequence_length) containing N sequences to make predictions for.
        :return: Tensor of predictions for each sequence of shape (N, 1).
        """

        inputs = inputs.type(torch.LongTensor)  # embedding requires long
        embeds = self.embedding(inputs)

        # # RNN
        # output, hn = self.rnn(embeds)

        # # LSTM
        # output, (hn, cn) = self.lstm(embeds)

        # RNN
        output, hn = self.gru(embeds)

        prediction = self.fc(hn.squeeze(0))

        return prediction

    def loss(self, logits, targets):
        """
        Computes the binary cross-entropy loss.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Binary cross entropy loss between logits and targets as a scalar tensor.
        """

        binary = torch.sigmoid(logits)
        targets = targets.type(torch.FloatTensor)  # cross entropy requires float
        loss = nn.functional.binary_cross_entropy(binary, targets)

        return loss

    def accuracy(self, logits, targets):
        """
        Computes the accuracy, i.e number of correct predictions / N.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Accuracy as a scalar tensor.
        """

        binary = torch.sigmoid(logits)
        prediction = torch.round(binary)

        accuracy = sum(prediction == targets).type(torch.FloatTensor)/len(targets)

        return accuracy


# Training parameters
TRAINING_BATCH_SIZE = 32
NUM_EPOCHS = 16
LEARNING_RATE = 1e-4

# Batch size for validation, this only affects performance.
VAL_BATCH_SIZE = 128
