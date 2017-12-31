
if __name__ == '__main__' and __package__ is None:
    import os
    os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "pytorch_bits"

import timeit

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_bits.nn import SRU


def test_cpu():
    a.detach_()
    b = cpu_sru(a)
    b.backward(loss)

def test_gpu():
    a.detach_()
    b = gpu_sru(a)
    b.backward(loss)

tests = ("test_cpu", "test_gpu",)

if __name__ == "__main__":
    for size in (50, 100, 200, 500, 1000):
        print("Size", size)
        try:
            a = Variable(torch.rand(size,size,size), requires_grad=True)
            loss = torch.ones_like(a)
            cpu_sru = SRU(size, 100, gpu=False)
            gpu_sru = SRU(size, 100, gpu=True)
            for test in tests:
                timer = timeit.Timer(test, globals=globals())
                print(test, np.mean(timer.repeat(number=1000000, repeat=10)))
        except RuntimeError:
            print("Not enough RAM. Aborting.")