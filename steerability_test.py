import argparse
import os
import torch
import numpy as np
from utils import *
from model import *
from vocab import Vocab
from batchify import get_batches2, get_batches, get_batches3

checkpoint_dir = "checkpoints/yelp/daae/"
parallel_data_dir = "parallel_data/"

vocab_file = os.path.join(checkpoint_dir, 'vocab.txt')
vocab = Vocab(vocab_file)

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}['aae'](vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

model = get_model(checkpoint_dir + "model.pt")
model.eval()

def encode(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        if args.enc == 'mu':
            zi = mu
        else:
            zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+args.batch_size], device=device)
        outputs = model.generate(zi, args.max_len, args.dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += args.batch_size
    return strip_eos(sents)

test_set_flag = False 
if (test_set_flag == True): 
    present_file = parallel_data_dir + "test.present"
    past_file = parallel_data_dir + "test.past"
else:
    present_file = parallel_data_dir + "present.txt"
    past_file = parallel_data_dir + "past.txt"

present_data = load_sent(present_file)
past_data = load_sent(past_file)

batch_size = 32
alpha = 1
data_batches, _ = get_batches2(present_data, past_data, vocab, batch_size, device)
word_batches, _ = get_batches3(present_data, past_data, vocab, batch_size, device)

def decode_logits(logits):
    word_inds = torch.argmax(logits, 2)
    words = [vocab.idx2word[ind] for ind in word_inds]
    return(words)

## load walk vector, and try
w = torch.load("walk.pt")
indices = list(range(len(data_batches)))
for i, idx in enumerate(indices):
    x_present = data_batches[idx][0]
    print("test set:", word_batches[idx][0])
    x_past = data_batches[idx][1]
    mu, logvar, z, logits = model(x_present)
    new_latent = z + alpha * w
    logits, hidden = model.decode(new_latent, x_present)
    words = decode_logits(logits)
    print("outcome:", words)


## arithmetic
'''
fa, fb, fc = "parallel_data/present.txt,parallel_data/past.txt,data/yelp/tense/test.present".split(',')
sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
za, zb, zc = encode(sa), encode(sb), encode(sc)
zd = zc + (zb.mean(axis=0) - za.mean(axis=0))
breakpoint()
sd = decode(zd)
write_sent(sd, "tests/arithmetic_present2past")
'''