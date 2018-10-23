import numpy as np
import torch, os
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import json

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

    def __init__(self, path, dictionary, min_freq=5, use_cuda=True, n_gram=5, create_dict=False, is_test=False):
        self.dictionary = dictionary
        self.min_freq = min_freq
        self.use_cuda = use_cuda
        self.data = []
        self.testing = {}
        self.tokenize(path, n_gram, create_dict, is_test)
        self.ignore_words(is_test)

    def test_tokenize(self, path, n_gram):
        with open(path, 'r') as f:
            for abstract_num, line in enumerate(f):
                j = json.loads(line.strip())
                generated = j["generated"]
                original = j["original"]
                words = generated.split()

                # Add SOS and EOS tokens
                words = [Corpus.SOS] + words + [Corpus.EOS]

                self.testing[abstract_num] = list(words)
                self.testing["org" + str(abstract_num)] = original.split()
                self.testing["gen" + str(abstract_num)] = list(words)

                # Generate n-gram training samples
                abstract_length = len(words)
                for i in range(0, abstract_length - n_gram):
                    X = words[i: i + n_gram]
                    Y = words[i + n_gram]
                    self.data.append((X, Y, abstract_num, i + n_gram))

    def tokenize(self, path, n_gram, create_dict, is_test):
        if is_test:
            self.test_tokenize(path, n_gram)
        else:
            with open(path, 'r') as f:
                for abstract_num, line in enumerate(f):
                    words = line.strip().split()
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

    def ignore_words(self, is_test):
        updated_data = []
        for d in self.data:
            X, Y = d[:2]
            if any(w not in self.dictionary.word2idx for w in X) or Y not in self.dictionary.word2idx:
                continue

            if any(self.dictionary.counter[w] < self.min_freq for w in X) or \
                self.dictionary.counter[Y] < self.min_freq:
                continue
            X = [self.dictionary.word2idx[w] for w in X]
            Y = self.dictionary.word2idx[Y]
            if not is_test:
                updated_data.append((X, Y))
            else:
                updated_data.append((X, Y, d[2], d[3]))
        self.data = updated_data

    def __getitem__(self, index):
        X, Y = self.data[index][:2]
        X = torch.LongTensor(X)
        Y = torch.LongTensor([Y])
        abstract_number, i = None, None

        # Implies testing mode
        if len(self.data[index]) > 2:
            abstract_number, i = self.data[index][2:]
            abstract_number = torch.LongTensor([abstract_number])
            i = torch.LongTensor([i])

        if self.use_cuda:
            X = X.cuda()
            Y = Y.cuda()
            # Necessary to use "is not None" with Tensors
            abstract_number = abstract_number.cuda() if abstract_number is not None else None
            i = i.cuda() if i is not None else None

        if abstract_number is not None:
            return X, Y, abstract_number, i
        return X, Y

    def __len__(self):
        return len(self.data)