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
    lr = 0.0001
    n_epochs = 20
    cell = "gru"
    n_gram = 5
    n_layers = 2
    hidden_size = 1150
    em_size = 512
    dropout_p = 0
    batch_size = 256
    max_grad_norm = 10
    log_interval = 1000
    patience = 5
    pre_trained = 'complete-512.vec'

parser = argparse.ArgumentParser(description='polishing networks')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mode', type=int,  default=0,
                    help='train(0)/predict_sentence(1)/predict_file(2) or evaluate(3)')
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
print("Configuration is as follows", json.dumps({"log_interval": config.log_interval,
                                                 "learning rate": config.lr,
                                                 "save": args.save,
                                                 "pre_trained": config.pre_trained,
                                                 "epochs": config.n_epochs,
                                                 "batch_size": config.batch_size,
                                                 "n-gram": config.n_gram,
                                                 "n-layers": config.n_layers,
                                                 "embedding size": config.em_size},sort_keys=True, indent=4, separators=(',', ': ')))

# Dictionary and corpus
dictionary = Dictionary()
training_corpus = Corpus(args.data+"/train.txt", dictionary, create_dict=True, use_cuda=args.cuda, n_gram=config.n_gram)
validation_corpus = Corpus(args.data+"/valid.txt", dictionary, create_dict=True, use_cuda=args.cuda, n_gram=config.n_gram)

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
    patience = 0
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
                  'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
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
        prev_best_metric = checkpoint['best_metric']
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

