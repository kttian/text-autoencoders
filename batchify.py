import torch

def get_batch(x, vocab, device):
    #breakpoint()
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)
    #print("ORDER", order[:5])
    #breakpoint()
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        single_batch = get_batch(data[i: j], vocab, device)
        batches.append(single_batch)
        #breakpoint()
        i = j
    return batches, order

def get_batch2(x, x_edited, vocab, device):
    #breakpoint()
    x_ret, xe_ret = [], []
    max_len = max([len(s) for s in x] + [len(s) for s in x_edited])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        x_ret.append([vocab.go] + s_idx + padding)

    for s in x_edited:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        xe_ret.append(s_idx + [vocab.eos] + padding)

    return torch.LongTensor(x_ret).t().contiguous().to(device), \
           torch.LongTensor(xe_ret).t().contiguous().to(device)  # time * batch

def get_batches2(data, edited, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data, edited), key=lambda i: len(i[1]))
    order, data, edited = zip(*z)
    #print("ORDER", order[:5])
    #breakpoint()
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        single_batch = get_batch2(data[i: j], edited[i: j], vocab, device)
        batches.append(single_batch)
        #breakpoint()
        i = j
    #print("batches shape", len(batches))
    #print(batches[0][0].shape)
    #print(batches[0][1].shape)
    return batches, order

def get_batches3(data, edited, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data, edited), key=lambda i: len(i[1]))
    order, data, edited = zip(*z)
    #print("ORDER", order[:5])
    #breakpoint()
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        single_batch = (data[i: j], edited[i: j])
        batches.append(single_batch)
        #breakpoint()
        i = j
    return batches, order
