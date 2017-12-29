import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import signalz

CLR = "\x1b[0K"

def get_batches(X, y, seq_len=100, reason=None):
    if seq_len > len(X): seq_len = len(X)
    batches = len(X) // seq_len
    leftover = len(X) % seq_len
    if reason == "training" and leftover > 0:
        offset = np.random.randint(leftover)
    else:
        offset = 0

    message = str(batches)
    if reason is not None:
        message += " " + reason
    message += " batches"
    if offset > 0:
        message += " @" + str(offset)

    start_time = time.time()
    for batch in range(batches):
        start = batch * seq_len
        end = start + seq_len
        yield batch, X[start:end], y[start:end]
        if batch + 1 < batches:
            print("\r%s -- %.1f%% done -- time %.5f ms/sample" % (message,
                100. * (batch + 1) / batches, 
                1000 * (time.time()-start_time) / (batch + 1) / (seq_len * X.size(1))), 
                end=CLR, flush=True)
    print("\r", end=CLR, flush=True)

def sine_1(length, pattern_length=60., add_noise=False, noise_range=(-0.1, 0.1)):
    X = np.arange(length) + np.random.randint(pattern_length)
    signal = np.sin(2 * np.pi * (X) / pattern_length)
    if add_noise:
        signal += np.random.uniform(noise_range[0], noise_range[1], size=signal.shape)
    return signal

def sine_2(length, pattern_length=60., add_noise=False, noise_range=(-0.1, 0.1)):
    X = np.arange(length) + np.random.randint(pattern_length)
    signal = (np.sin(2 * np.pi * (X) / pattern_length) + np.sin(2 * 2 * np.pi * (X) / pattern_length)) / 2.0
    if add_noise:
        signal += np.random.uniform(noise_range[0], noise_range[1], size=signal.shape)
    return signal

def sine_3(length, pattern_length=60., add_noise=False, noise_range=(-0.1, 0.1)):
    X = np.arange(length) + np.random.randint(pattern_length)
    signal = (np.sin(2 * np.pi * (X) / pattern_length) + np.sin(2 * 2 * np.pi * (X) / pattern_length) + np.sin(2 * 3 * np.pi * (X) / pattern_length)) / 3.0
    if add_noise:
        signal += np.random.uniform(noise_range[0], noise_range[1], size=signal.shape)
    return signal

def mackey_glass(length, add_noise=False, noise_range=(-0.01, 0.01)):
    initial = .25 + .5 * np.random.rand()
    signal = signalz.mackey_glass(length, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=initial)
    if add_noise:
        signal += np.random.uniform(noise_range[0], noise_range[1], size=signal.shape)
    return signal - 1.

def levy_flight(length, add_noise=False, noise_range=(-0.01, 0.01)):
    offset = np.random.randint(length // 2)
    signal = signalz.levy_flight(length + offset, alpha=1.8, beta=0., sigma=.01, position=0)
    return signal[offset:] - 1.

def brownian(length, add_noise=False, noise_range=(-0.01, 0.01)):
    return signalz.brownian_noise(length, leak=0.1, start=0, std=.1, source="gaussian")

generators = {
    "sine_1": sine_1,
    "sine_2": sine_2,
    "sine_3": sine_3,
    "mackey_glass": mackey_glass,
    "levy_flight": levy_flight,
    "brownian": brownian,
}


def generate_data(data_fn="sine_1", length=10000, pattern_length=60., batch_size=32, add_noise=False):
    X = np.empty((length, batch_size, 1))
    y = np.empty((length, batch_size, 1))
    
    for b in range(batch_size):
        x_data = generators[data_fn](length + 1, add_noise=add_noise)
        
        if b == 0:
            plt.figure("Synthetic data", figsize=(15, 10))
            plt.title("Synthetic data")
            plt.plot(range(min(1000, length)), x_data[:min(1000, length)])
        
        X[:,b,0] = x_data[:-1]
        y[:,b,0] = x_data[1:]

    plt.savefig("synthetic_data.png")
    plt.close()

    # 70% training, 10% validation, 20% testing
    train_sep = int(length * 0.7)
    val_sep = train_sep + int(length * 0.1)
    
    X_train = Variable(torch.from_numpy(X[:train_sep, :]).float(), requires_grad=False)
    y_train = Variable(torch.from_numpy(y[:train_sep, :]).float(), requires_grad=False)
    
    X_val = Variable(torch.from_numpy(X[train_sep:val_sep, :]).float(), requires_grad=False)
    y_val = Variable(torch.from_numpy(y[train_sep:val_sep, :]).float(), requires_grad=False)
    
    X_test = Variable(torch.from_numpy(X[val_sep:, :]).float(), requires_grad=False)
    y_test = Variable(torch.from_numpy(y[val_sep:, :]).float(), requires_grad=False)
    
    print(("X_train size = {}, X_val size = {}, X_test size = {}".format(X_train.size(), X_val.size(), X_test.size())))       
    
    return X_train, X_val, X_test, y_train, y_val, y_test
