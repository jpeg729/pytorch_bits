
if __name__ == '__main__' and __package__ is None:
    import os
    os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "pytorch_bits"

import timeit

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import pytorch_bits.activations as activations


def test_tanh():
    a.detach_()
    b = nn.Tanh(a)
    b.backward(loss)

def test_hardtanh():
    a.detach_()
    b = nn.HardTanh(a)
    b.backward(loss)

def test_sigmoid():
    a.detach_()
    b = nn.Sigmoid(a)
    b.backward(loss)

def test_elu():
    a.detach_()
    b = nn.ELU(a)
    b.backward(loss)

def test_isrlu():
    a.detach_()
    b = activations.ISRLU(a)
    b.backward(loss)

def test_isru_tanh():
    a.detach_()
    b = activations.ISRU_tanh(a)
    b.backward(loss)

def test_isru_sigmoid():
    a.detach_()
    b = activations.ISRU_sigmoid(a)
    b.backward(loss)

def test_isru_softsign():
    a.detach_()
    b = nn.SoftSign(a)
    b.backward(loss)

tests = ("test_tanh", "test_hardtanh", "test_sigmoid", 
    "test_isru_tanh", "test_isru_sigmoid", "test_isru_softsign",
    "test_elu", "test_isrlu", 
    )

if __name__ == "__main__":
    for size in (50, 100, 200, 500, 1000):
        print("Size", size)
        try:
            a = Variable(torch.rand(size,size,size), requires_grad=True)
            loss = torch.ones_like(a)
            for test in tests:
                timer = timeit.Timer(test, globals=globals())
                print(test, np.mean(timer.repeat(number=1000000, repeat=5)))
        except RuntimeError:
            print("Not enough RAM. Aborting.")