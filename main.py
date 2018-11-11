import time, argparse, torch
from torch.backends import cudnn
from model import LSTM_LM
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from data import Corpus, Dictionary
from torch.utils.data import DataLoader
from data import load_embeddings
import random, os, json

class Config:
    def __init__(self):
        self.lr = 0.00001
        self.n_epochs = 50
        self.cell = "gru"
        self.n_gram = 5
        self.n_layers = 2
        self.hidden_size = 1150
        self.em_size = 512
        self.dropout_p = 0
        self.batch_size = 256
        self.max_grad_norm = 10
        self.log_interval = 1000
        self.patience = 5
        self.pre_trained = 'complete-512.vec'
        self.context_mode = "default"
        self.bidirectional = False

    def __repr__(self):
        return "Configuration is as follows {}".format(json.dumps({"log_interval": self.log_interval,
                                                 "cell": self.cell,
                                                 "bidirectional": self.bidirectional,
                                                 "learning rate": self.lr,
                                                 "save": args.save,
                                                 "pre_trained": self.pre_trained,
                                                 "epochs": self.n_epochs,
                                                 "batch_size": self.batch_size,
                                                 "n-gram": self.n_gram,
                                                 "n-layers": self.n_layers,
                                                 "embedding size": self.em_size,
                                                 "context mode": self.context_mode},sort_keys=True, indent=4, separators=(',', ': ')))


parser = argparse.ArgumentParser(description='polishing networks')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mode', type=int,  default=0,
                    help='train(0)/predict_file(1)')
parser.add_argument('--data', type=str,
                    help="Data location")
parser.add_argument('--save', type=str,
                    help='Location of checkpoint file')

args = parser.parse_known_args()[0]
cudnn.benchmark = True

# Set the seeds for reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Config to run
config = Config()
if os.path.isfile(args.save):
    checkpoint = torch.load(args.save)
    if 'config' in checkpoint:
        print("Loading saved config")
        config = checkpoint['config']
print(config)

# Dictionary and corpus
dictionary = Dictionary()
training_corpus = Corpus(args.data+"/train.txt", dictionary, create_dict=True, use_cuda=args.cuda, n_gram=config.n_gram, context_mode=config.context_mode)
validation_corpus = Corpus(args.data+"/valid.txt", dictionary, create_dict=True, use_cuda=args.cuda, n_gram=config.n_gram, context_mode=config.context_mode)

# TensorboardX object
writer = SummaryWriter("saved_runs/" + args.save)

# Word embeddings
embedding = nn.Embedding(len(dictionary), config.em_size, padding_idx=0)
if config.pre_trained:
    load_embeddings(embedding, dictionary.word2idx, config.pre_trained, config.em_size)

# Model, Optimizer and Loss
model = LSTM_LM(embedding, config)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Model total parameters:', total_params, flush=True)

def train(start_epoch, best_metric):
    train_loader = DataLoader(training_corpus, config.batch_size)
    patience = config.patience
    training_loss = AverageMeter()

    try:
        for epoch in range(start_epoch, config.n_epochs + 1):

            # Put the model in training mode.
            model.train(True)

            start = time.time()
            epoch_time = time.time()
            train_time = AverageMeter()

            print("----------- Start Processing Epoch {}".format(epoch), flush=True)
            for batch_idx, data in enumerate(train_loader):
                X = data[0]
                Y = data[1]
                output, target = model(X, Y)
                loss = criterion(output, target)
                training_loss.update(loss.item(), X.size(0))

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                # Reset time for next batch and record current batch's time.
                train_time.update((time.time() - start) * 1000)
                start = time.time()

                if batch_idx and batch_idx % config.log_interval == 0:
                    print("Processed {:5.2f}% examples, training_loss {:5.2f}, ms/batch {:5.2f}".format(
                        (batch_idx * X.size(0)) / len(train_loader.dataset), training_loss.avg, train_time.avg
                    ), flush=True)

            validation_loss = evaluate()
            writer.add_scalar('loss/train', training_loss.avg, epoch)
            writer.add_scalar('loss/valid', validation_loss.avg, epoch)

            print("End of epoch: {}, train_loss: {:5.2f}, validation_loss: {:5.2f}, total_time = {:5.2f}s".format(epoch, training_loss.avg,
                                                                                 validation_loss.avg, time.time() - epoch_time), flush=True)

            if validation_loss.avg >= best_metric:
                patience -= 1

                if patience == 0:
                    print("Breaking off now. Performance has not improved on validation set since the last",
                          config.patience, "epochs", flush=True)
                    exit(0)

            else:
                patience = config.patience
                best_metric = validation_loss.avg
                print("Saving best model!", flush=True)
                save(epoch, best_metric)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def evaluate():
    valid_loader = DataLoader(validation_corpus, config.batch_size)
    validation_loss = AverageMeter()
    model.eval()
    for batch_idx, data in enumerate(valid_loader):
        X = data[0]
        Y = data[1]
        output, target = model(X, Y)
        loss = criterion(output, target)
        validation_loss.update(loss.item(), X.size(0))
    return validation_loss


def save(epoch, metric):
    checkpoint = {"model" : model.state_dict(), "best_metric" : metric,
                  'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'config': config}
    torch.save(checkpoint, args.save)

def load():
    start_epoch = 0
    prev_best_metric = float("inf")
    if os.path.isfile(args.save):
        print("=> loading checkpoint '{}'".format(args.save), flush=True)
        checkpoint = torch.load(args.save)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_metric = checkpoint['best_metric']
        print("=> loaded best model with metric {}, epoch {}".format(best_metric, start_epoch), flush=True)
    else:
        print("=> no checkpoint found at '{}', starting from scratch".format(args.save), flush=True)
    return start_epoch, prev_best_metric

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    start_epoch, best_metric = load()
    if args.mode == 0:
        train(start_epoch, best_metric)
    elif args.mode == 1:
        test_corpus = Corpus(args.data + "/generated.txt", dictionary, use_cuda=args.cuda,
                             n_gram=config.n_gram, is_test=True)
        test_loader = DataLoader(test_corpus, config.batch_size)
        print("Number of test samples", len(test_loader.dataset))
        softmax = nn.Softmax(dim=1)
        changes_made = 0
        thinks_same = 0
        for batch_idx, (X, Y, abstract_number, i) in enumerate(test_loader):
            output, target = model(X, Y)
            output = softmax(output)
            probabilities, words = output.topk(1)
            for org, gen, prob, ab, index in zip(Y, words, probabilities, abstract_number, i):
                # Highly confident about the new word
                if prob > 0.5:
                    changes_made += (org != gen).item()
                    thinks_same += (org == gen).item()
                    new_word = test_corpus.dictionary.idx2word[gen.item()]
                    test_corpus.testing[ab.item()][index.item()] = new_word

        with open("polished.txt", "w") as f:
            for abstract in test_corpus.testing:
                if type(abstract) == int:

                    original = test_corpus.testing["org"+str(abstract)]
                    org_generated = test_corpus.testing["gen"+str(abstract)][1:-1]
                    polished = test_corpus.testing[abstract][1:-1]
                    differences = []
                    for i, (w1, w2) in enumerate(zip(org_generated, polished)):
                        if w1 != w2:
                            differences.append((i, w1, w2))

                    f.write(json.dumps({"original": " ".join(original),
                                    "generated": " ".join(polished),
                                    "before polishing": " ".join(org_generated),
                                    "differences": differences}))
                    f.write("\n")
        print("Changes made:", changes_made)
        print("Done!")
