import sys
import time
from argparse import ArgumentParser
import numpy as np
import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import data_generation
import models
# from pytorch_custom.yellowfin import YFOptimizer

parser = ArgumentParser(description='PyTorch example')
parser.add_argument('--data_fn', type=str, default="sine_3")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--length', type=int, default=1000)
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--layers', type=str, nargs="+", default=["LSTM_51"])
parser.add_argument('--sigmoid', type=str, default=None)
parser.add_argument('--tanh', type=str, default=None)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
print(args)

X_train, X_val, X_test, y_train, y_val, y_test = data_generation.generate_data(
    data_fn=args.data_fn, batch_size=args.batch_size, 
    length=args.length, add_noise=args.add_noise)

rnn = models.Model(input_size=X_train.size(-1), layers=args.layers, output_size=y_train.size(-1),
    sigmoid=args.sigmoid, tanh=args.tanh)
print(rnn)
print(sum([p.numel() for p in rnn.parameters() if p.requires_grad]), "trainable parameters")

loss_fn = nn.MSELoss()
optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9)
# optimizer = optim.SGD(rnn.parameters(), lr=0.000003, momentum=0.95)
# optimizer = optim.LBFGS(rnn.parameters())
# optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
# optimizer = YFOptimizer(rnn.parameters())
# optimizer = optim.Adagrad(rnn.parameters(), lr=0.0001)
"""
Training with ground truth -- The input is the ground truth
"""
try:
    val_loss_list = []
    start = time.time()
    for epoch in range(args.epochs):
        training_loss = 0
        rnn.train(True)
        rnn.reset_hidden()
        for batch, data, target in data_generation.get_batches(X_train, y_train, seq_len=args.seq_len, reason="training"):
            output = rnn(data)
            optimizer.zero_grad()
            if batch == 0:
                loss = loss_fn(output[args.warmup:], target[args.warmup:])
            else:
                loss = loss_fn(output, target)
            if args.verbose:
                print("Input:  mean", data.mean().data[0],   "std", data.std().data[0])
                print("Output: mean", output.mean().data[0], "std", output.std().data[0])
                print("Target: mean", target.mean().data[0], "std", target.std().data[0])
            loss.backward(retain_graph=True)
            if batch > 0:
                optimizer.step()
                training_loss += loss.data[0]
            rnn.detach_hidden()
        training_loss /= batch + 1

        val_loss = 0
        rnn.train(False)
        rnn.reset_hidden()
        for batch, data, targets in data_generation.get_batches(X_val, y_val, seq_len=args.seq_len, reason="validation"):
            output = rnn(data)
            loss = loss_fn(output, targets)
            val_loss += loss.data[0]
        val_loss /= batch + 1
        val_loss_list.append(val_loss)
        print("Ground truth - Epoch " + str(epoch) + " -- train loss = " + str(training_loss) + " -- val loss = " + str(val_loss)
            + " -- time %.1fs" % ((time.time() - start) / (epoch + 1)))

        # Early stopping condition -- when the last 4 epochs results in a validation error < 0.015
        if len(val_loss_list) >= 4 and max(val_loss_list[-4:]) < 0.015:
            print("Triggering early stopping criteria")
            break
except KeyboardInterrupt:
    print("\nTraining interrupted")

"""
Measuring the test score -> running the test data on the model
"""
rnn.train(False)
rnn.reset_hidden()
test_loss = 0
list1 = []
list2 = []
for batch, data, targets in data_generation.get_batches(X_test, y_test, seq_len=args.seq_len, reason="testing"):
    output = rnn(data)
    loss = loss_fn(output, targets)
    test_loss += loss.data[0]
    target_last_point = torch.squeeze(targets[:, -1]).data.cpu().numpy().tolist()
    pred_last_point = torch.squeeze(output[:, -1]).data.cpu().numpy().tolist()
    list1 += target_last_point
    list2 += pred_last_point
    if len(list1) > 400:
        break
plt.figure(1)
plt.plot(list1, "b")
plt.plot(list2, "r")
plt.legend(["Original data", "Generated data"])
test_loss /= batch + 1
print("Test loss = ", test_loss)

if epoch > 0:
    plt.show()
