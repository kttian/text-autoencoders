import argparse
from meter import AverageMeter
import os
import torch
import numpy as np
from utils import *
from model import *
from vocab import Vocab
from batchify import get_batches2, get_batches, get_batches3
import time
from train import evaluate
import matplotlib.pyplot as plt
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--pres', metavar='FILE', required=True,
                    help='name of present data file')
parser.add_argument('--past', metavar='FILE', required=True,
                    help='name of past data file')
parser.add_argument('--walk_file', metavar='FILE', required=True,
                    help='name of walk pt file')
# parser.add_argument('--results_dir', metavar='FILE', required=True,
#                     help='name of walk pt file')
parser.add_argument('--init_mode', default="rand", metavar='P',
                    help='mode for initializing w')
parser.add_argument('--num_epochs', type=int, default=100, metavar='P',
                    help='number of epochs for training w')
parser.add_argument('--eval', type=bool, default=False, metavar='P',
                    help='True if evaluating, False if training')
parser.add_argument('--verbose', type=bool, default=False, metavar='P',
                    help='True for verbose printouts, False o/w')
##########################################################################
checkpoint_dir = "checkpoints/yelp/daae3/"
parallel_data_dir = ""
results_dir = "results_0613_joint/"
print_outputs_flag = False 

# parameters
set_seed(1111)
# pres_fn = "present.txt"
# past_fn = "past.txt"
# walk_file = "walk_test.pt"
# init_mode = "rand"
# num_epochs = 100

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
    return model, train_args

def load_model(checkpoint_dir, verbose = False):
    model, train_args = get_model(checkpoint_dir + "model.pt")
    if verbose:
        print("MODEL")
        print(model)

    model.train()
    return model, train_args

def load_data(pres_fn, past_fn):
    present_file = parallel_data_dir + pres_fn
    past_file = parallel_data_dir + past_fn
    present_data = load_sent(present_file)
    past_data = load_sent(past_file)
    n_sents = len(present_data)
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
        w = torch.load("walk_files/arithmetic_v2.pt")
        w = w.to(device)
        w.requires_grad = True
    return w

def compute_loss(w, x, x_edit, model):
    mu, logvar = model.encode(x)
    z = reparameterize(mu, logvar)
    new_latent = z + alpha * w
    logits, hidden = model.decode(new_latent, x)
    loss = model.loss_rec(logits, x_edit).mean()
    return loss

def average_loss(w, data_batches, model, verbose = False):
    meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        total_loss = 0
        B = len(data_batches)
        nsents = 0
        for idx in range(len(data_batches)):
            x, x_edit = data_batches[idx]
            
            mu, logvar = model.encode(x)
            z = reparameterize(mu, logvar)
            new_latent = z + alpha * w
            logits, hidden = model.decode(new_latent, x)
            loss = model.loss_rec(logits, x_edit).mean()

            if verbose:
                losses = model.autoenc(x, x_edit)
                print("autoenc", idx, ":", losses['rec'], " | shapes", x.shape, x_edit.shape)
                print("my loss", idx, ":", loss)
                # print("x", x.shape, "| x_edit", x_edit.shape)
            if print_outputs_flag:
                sents = []
                edited_sents = []
                walk_sents = []
                batch_len = x.shape[1]

                max_len = 35
                dec = 'greedy'
                outputs = model.generate(new_latent, max_len, dec).t()
                for i in range(batch_len):
                    x_i = x[:,i]
                    sents.append([vocab.idx2word[id] for id in x_i])
                    xe_i = x_edit[:,i]
                    edited_sents.append([vocab.idx2word[id] for id in xe_i])
                    output_i = outputs[i]
                    walk_sents.append([vocab.idx2word[id] for id in output_i])
                    walk_sents = strip_eos(walk_sents)

                for i in range(batch_len):
                    x_i = torch.unsqueeze(x[:,i], dim=1)
                    xe_i = torch.unsqueeze(x_edit[:,i], dim=1)
                    loss_i = compute_loss(w, x_i, xe_i, model)
                    print("batch", idx, ":", loss, "| sentence", i, ":", loss_i)
                    print("--SENT:", sents[i])
                    print(x[:,i])
                    print("--EDIT:", edited_sents[i])
                    print(x_edit[:,i])
                    print("--WALK:", walk_sents[i])
                    print(outputs[i])

            total_loss += loss * x.shape[1]
            nsents += x.shape[1]
            #breakpoint()
            meter.update(loss.item(), x.shape[1])
        
        avg_loss = total_loss/nsents
        if verbose:
            print("avg_loss meter loss vs avg_loss", meter.avg, avg_loss)
        #print("average loss", avg_loss)
        #print("=" * 60)
    return avg_loss

def plot_series(series_list, walk_file):
    for series in series_list:
        plt.plot(series[1])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    print("saved figure", results_dir+walk_file[:-2]+"png")
    plt.savefig(results_dir+walk_file[:-2]+"png")

def train_walk(walk_file, w, data_batches, valid_batches, model, num_epochs, verbose = False):
    # for param in model.parameters():
    #     param.requires_grad = False # freeze the model
    print("START TRAINING:", walk_file)
    joint_walk = itertools.chain([w], model.parameters())
    opt = optim.Adam(joint_walk, lr=0.0005)
    # opt = optim.Adam([w], lr=0.01, momentum=0.9)
    start_time = time.perf_counter()
    meter = AverageMeter()

    loss_hist_before = []
    loss_hist_during = []
    for e in range(num_epochs):
        avg_loss_before = average_loss(w, data_batches, model, verbose)
        model.train()
        total_loss = 0
        nsents = 0
        meter.clear()
        indices = list(range(len(data_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            opt.zero_grad()
            x, x_edit = data_batches[idx]

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
            total_loss += loss * x.shape[1]
            nsents += x.shape[1]
            meter.update(loss, x.shape[1])

        print("---------------------------")
        avg_loss_after = average_loss(w, data_batches, model)
        print("FINISHED EPOCH", e)
        print("avg loss before:", avg_loss_before)
        print("avg train loss: ", total_loss/nsents)
        # print("meter loss", meter.avg)
        loss_hist_before.append((e, avg_loss_before.item()))
        loss_hist_during.append((e, meter.avg.item()))
        if verbose:
            print("loss", loss)
            print("nsents", nsents)
        val_loss = average_loss(w, valid_batches, model, False)
        print("avg valid loss: ", val_loss)
        epoch_time = time.perf_counter()
        print("time: ", epoch_time - start_time)
        print("=" * 60)

    print("FINISHED TRAINING")
    best_before_loss = min(loss_hist_before, key = lambda x : x[1])
    best_during_loss = min(loss_hist_during, key = lambda x : x[1])
    print("best_before_loss:", best_before_loss, loss_hist_during[best_before_loss[0]])
    print("best_during_loss:", best_during_loss, loss_hist_before[best_during_loss[0]])
    plot_series([loss_hist_before, loss_hist_during], walk_file)
    print(w)

    ### save walk and model
    torch.save(w, os.path.join(results_dir, walk_file))
    return w, model

#print(total_loss)
def evaluation(pres_fn, past_fn, w, model, verbose):
    batches = load_data(pres_fn, past_fn)
    return average_loss(w, batches, model, verbose)

def main(args):
    pres_fn = args.pres
    past_fn = args.past
    walk_file = args.walk_file
    init_mode = args.init_mode 
    num_epochs = args.num_epochs
    val_pres_fn = "parallel_data/test_present.txt"
    val_past_fn = "parallel_data/test_past.txt"
    verbose = args.verbose 

    model, train_args = load_model(checkpoint_dir)
    train_batches = load_data(pres_fn, past_fn)
    valid_batches = load_data(val_pres_fn, val_past_fn)

    if verbose:
        print("~" * 50)
        meters = evaluate(model, train_batches)
        print(' '.join(['{} {:.2f},'.format(k, meter.avg)
                for k, meter in meters.items()]))
        print("~" * 50)
    
    if args.eval: #evaluate only
        w_final = torch.load(results_dir + walk_file, device)
        print("EVALUATION")
        print("train loss")
        average_loss(w_final, train_batches, model, verbose)
        print("-" * 40)
        print("valid loss")
        average_loss(w_final, valid_batches, model, verbose)
        print("-" * 40)
    else:
        # prints
        print("walk_file: \t", walk_file)
        print("data files: \t", pres_fn, ",", past_fn)
        print("init mode: \t", init_mode)
        print("num epochs: \t", num_epochs)
        print("-" * 60)
        
        # initial
        w = initialize(init_mode)
        print("INITIAL LOSS")
        init_loss = average_loss(w, train_batches, model, verbose)
        print("initial avg loss", init_loss)
        print("-" * 40)
        # training
        w_final, model_final = train_walk(walk_file, w, train_batches, valid_batches, 
                                          model, num_epochs, verbose)
        
        ckpt = {'args': train_args, 'model': model_final.state_dict()}
        torch.save(ckpt, os.path.join(results_dir, 'joint_model.pt'))

        # evaluation
        print("EVALUATION")
        print("evaluating on training data")
        valid_pres_file = pres_fn # "test_present.txt" usually
        valid_past_file = past_fn
        valid_batches = load_data(valid_pres_file, valid_past_file)
        final_loss = average_loss(w_final, valid_batches, model, verbose)
        print(final_loss)

        print("\n LOSS DETAILS")
    #else: # train to max valid loss


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
