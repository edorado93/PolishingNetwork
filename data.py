import numpy as np
import torch, os
from gensim.models import KeyedVectors
from torch.utils.data import Dataset

def load_embeddings(pytorch_embedding, word2idx, filename, embedding_size):
    print("Copying pretrained word embeddings from ", filename, flush=True)
    en_model = KeyedVectors.load_word2vec_format(filename)
    pretrained_words = set()
    for word in en_model.vocab:
        pretrained_words.add(word)

    arr = [0] * len(word2idx)
    for word in word2idx:
        index = word2idx[word]
        if word in pretrained_words:
            arr[index] = en_model[word]
        else:
            arr[index] = np.random.uniform(-1.0, 1.0, embedding_size)

    arr = np.array(arr)
    pytorch_embedding.weight.data.copy_(torch.from_numpy(arr))
    return pytorch_embedding

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = {}
        self.add_word(Corpus.SOS)
        self.add_word(Corpus.EOS)
        self.add_word(Corpus.UNK)
        self.add_word(Corpus.PAD)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        self.counter[word] = self.counter.get(word, 0) + 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(Dataset):

    UNK = '<UNK>'
    PAD = '<PAD>'
    EOS = '<EOS>'
    SOS = '<SOS>'

    def __init__(self, path, dictionary, min_freq=5, use_cuda=True, n_gram=5, create_dict=False):
        self.dictionary = dictionary
        self.min_freq = min_freq
        self.use_cuda = use_cuda
        self.data = []
        self.tokenize(path, n_gram, create_dict)
        self.ignore_words()

    def tokenize(self, path, n_gram, create_dict):
        with open(path, 'r') as f:
            for line in f:
                words = line.split()

                if create_dict:
                    # Add the words of the current abstract to the dictionary
                    for word in words:
                        self.dictionary.add_word(word)

                # Add SOS and EOS tokens
                words = [Corpus.SOS] + words + [Corpus.EOS]

                # Generate n-gram training samples
                abstract_length = len(words)
                for i in range(0, abstract_length - n_gram):
                    X = words[i : i + n_gram]
                    Y = words[i + n_gram]
                    self.data.append((X, Y))

    def ignore_words(self):
        updated_data = []
        for X, Y in self.data:
            # Ignore samples with UNK in them.
            if any(self.dictionary.counter[w] < self.min_freq for w in X):
                continue
            X = [self.dictionary.word2idx[w] for w in X]
            Y = self.dictionary.word2idx[Y]
            updated_data.append((X, Y))
        self.data = updated_data

    def __getitem__(self, index):
        X, Y = self.data[index]
        X = torch.LongTensor(X)
        Y = torch.LongTensor([Y])

        if self.use_cuda:
            X = X.cuda()
            Y = Y.cuda()

        return X, Y

    def __len__(self):
        return len(self.data)