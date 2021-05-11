import argparse
import os
import torch
import numpy as np
from utils import *
from model import *
from vocab import Vocab
from batchify import get_batches2, get_batches, get_batches3
import time

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--pres', metavar='FILE', required=True,
                    help='name of present data file')
parser.add_argument('--past', metavar='FILE', required=True,
                    help='name of past data file')
parser.add_argument('--walk_file', metavar='FILE', required=True,
                    help='name of walk pt file')
parser.add_argument('--init_mode', default="rand", metavar='P',
                    help='mode for initializing w')
parser.add_argument('--num_epochs', type=int, default=100, metavar='P',
                    help='number of epochs for training w')

##########################################################################
checkpoint_dir = "checkpoints/yelp/daae/"
parallel_data_dir = "parallel_data/"
results_dir = "results/"

# parameters
set_seed(1111)
pres_fn = "present.txt"
past_fn = "past.txt"
walk_file = "walk_test.pt"
init_mode = "rand"
num_epochs = 100


##########################################################################
vocab_file = os.path.join(checkpoint_dir, 'vocab.txt')
#if not os.path.isfile(vocab_file):
#    Vocab.build(train_sents, vocab_file, args.vocab_size)
vocab = Vocab(vocab_file)

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# hyper parameters 
alpha = 1
dim_emb = 128 
batch_size = 256


def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}['aae'](vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

def encode(sents):
    batches, order = get_batches(sents, vocab, batch_size, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def get_arithmetic(pres_data, past_data):
    za, zb = encode(pres_data), encode(past_data)
    w = torch.tensor(zb.mean(axis=0) - za.mean(axis=0), requires_grad = True, device=device)
    return w

def load_model(checkpoint_dir):
    model = get_model(checkpoint_dir + "model.pt")
    
    # model.train()
    # for param in model.parameters():
    #     param.requires_grad = False # freeze the model

    return model

def load_data(pres_fn, past_fn):
    present_file = parallel_data_dir + pres_fn
    past_file = parallel_data_dir + past_fn
    present_data = load_sent(present_file)
    past_data = load_sent(past_file)
    print("present, past data length:", len(present_data), len(past_data))
    data_batches, _ = get_batches2(present_data, past_data, vocab, batch_size, device)
    B = len(data_batches)
    print("number of batches:", B)
    return data_batches

def initialize(init_mode):
    if init_mode == "rand":
        w = torch.randn(dim_emb, requires_grad=True, device=device)
    elif init_mode == "zero":
        w = torch.zeros(dim_emb, requires_grad=True, device=device)
    elif init_mode == "arithmetic":
        w = torch.load(results_dir + "arithmetic.pt")
        w.requires_grad = True
        w = w.to(device)
    return w

def print_inital_loss(init_mode, w, data_batches, model):
    print("INITIAL LOSS:", init_mode, "init")
    total_loss = 0
    B = len(data_batches)
    for idx in range(len(data_batches)):
        x = data_batches[idx][0]
        x_edit = data_batches[idx][1]
        
        mu, logvar = model.encode(x)
        z = reparameterize(mu, logvar)
        new_latent = z + alpha * w
        logits, hidden = model.decode(new_latent, x)
        loss = model.loss_rec(logits, x_edit).mean()
        print("LOSS", idx, ":", loss)
        total_loss += loss

    print("average loss", total_loss/B)
    print("=" * 60)

def train_walk(walk_file, w, data_batches, model, num_epochs):
    print("START TRAINING:", walk_file)
    opt = optim.SGD([w], lr=0.01, momentum=0.9)
    start_time = time.perf_counter()
    B = len(data_batches)
    for e in range(num_epochs):
        total_loss = 0
        indices = list(range(len(data_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            opt.zero_grad()
            x = data_batches[idx][0]
            x_edit = data_batches[idx][1]
            #print(x_edit[0])
            #print(x_edit_one_hot[0])

            # encode the input x
            mu, logvar = model.encode(x)
            z = reparameterize(mu, logvar)
            # add w to compute new latent
            new_latent = z + alpha * w
            # decode the new latent
            logits, hidden = model.decode(new_latent, x)
            # compute the loss wrt to the edit
            loss = model.loss_rec(logits, x_edit).mean()
            #print("LOSS", idx, ":", loss)

            loss.backward()
            opt.step()
            total_loss += loss

        print("---------------------------")
        print("FINISHED EPOCH", e)
        print("average loss", total_loss/B)
        print("loss", loss)
        epoch_time = time.perf_counter()
        print("time", epoch_time)
        print("=" * 60)

    print("FINISHED TRAINING")
    print(w)
    torch.save(w, results_dir + walk_file)
    return w 

#print(total_loss)

def main(args):
    pres_fn = args.pres
    past_fn = args.past
    walk_file = args.walk_file
    init_mode = args.init_mode 
    num_epochs = args.num_epochs
    
    # prints
    print("walk_file: \t", walk_file)
    print("data files: \t", pres_fn, ",", past_fn)
    print("init mode: \t", init_mode)
    print("num epochs: \t", num_epochs)
    print("-" * 60)

    model = load_model(checkpoint_dir)
    data_batches = load_data(pres_fn, past_fn)
    w = initialize(init_mode)
    print_inital_loss(init_mode, w, data_batches, model)
    w_final = train_walk(walk_file, w, data_batches, model, num_epochs)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
