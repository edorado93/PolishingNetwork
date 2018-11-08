import torch.nn as nn

class LSTM_LM(nn.Module):
    def __init__(self, embedding, config):
        super(LSTM_LM, self).__init__()
        self.embedding = embedding
        self.vocab_size = embedding.num_embeddings
        if config.cell == "gru":
            self.rnn = nn.GRU(config.em_size, config.hidden_size, config.n_layers,
                              batch_first=True, dropout=config.dropout_p, bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(config.em_size, config.hidden_size, config.n_layers,
                              batch_first=True, dropout=config.dropout_p, bidirectional=config.bidirectionals)

        self.out = nn.Linear(2 * config.hidden_size if config.bidirectional else config.hidden_size, self.vocab_size)

    def forward(self, X, Y):
        embedded = self.embedding(X)
        output, hidden = self.rnn(embedded)

        # Only consider the final output.
        vocab_output = self.out(output[:, -1, :])

        # Transform tensors for loss computation
        vocab_output = vocab_output.reshape(-1, self.vocab_size)
        Y = Y.reshape(-1)

        return vocab_output, Y

