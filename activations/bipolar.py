import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Bipolar(nn.Module):
    def __init__(self, activation, input_size):
        super(Bipolar, self).__init__()
        self.activation = activation
        self.positive_indices = Variable(torch.Tensor([0]*input_size), requires_grad=False)
        for i in range(0, input_size, 2):
            self.positive_indices[i] = 1

    def forward(self, data):
        pos_output = self.activation(data) * self.positive_indices
        neg_output = -self.activation(-data) * (1 - self.positive_indices)
        return pos_output + neg_output


if __name__ == "__main__":
    import numpy as np

    test_size = 10
    data = Variable(torch.randn(5,5,test_size))

    print("Testing Bipolar(relu)", end=" ")
    activation = F.relu
    bipolar = Bipolar(activation, test_size)
    out = bipolar(data)
    failed = False
    for i in range(test_size):
        if i % 2 == 0:
            if not np.allclose(out[:, :, i].data.numpy(), activation(data[:, :, i]).data.numpy()):
                print("\nfailed at index ", i)
                failed = True
        else:
            if not np.allclose(out[:, :, i].data.numpy(), -activation(-data[:, :, i]).data.numpy()):
                print("\nfailed at index ", i)
                failed = True
    if not failed:
        print("-- passed")

    print("Testing Bipolar(elu)", end=" ")
    activation = F.elu
    bipolar = Bipolar(activation, test_size)
    out = bipolar(data)
    failed = False
    for i in range(test_size):
        if i % 2 == 0:
            if not np.allclose(out[:, :, i].data.numpy(), activation(data[:, :, i]).data.numpy()):
                print("\nfailed at index ", i)
                failed = True
        else:
            if not np.allclose(out[:, :, i].data.numpy(), -activation(-data[:, :, i]).data.numpy()):
                print("\nfailed at index ", i)
                failed = True
    if not failed:
        print("-- passed")

    print("Testing Bipolar(sigmoid)", end=" ")
    activation = F.sigmoid
    bipolar = Bipolar(activation, test_size)
    out = bipolar(data)
    failed = False
    for i in range(test_size):
        if i % 2 == 0:
            if not np.allclose(out[:, :, i].data.numpy(), activation(data[:, :, i]).data.numpy()):
                print("\nfailed at index ", i)
                failed = True
        else:
            if not np.allclose(out[:, :, i].data.numpy(), -activation(-data[:, :, i]).data.numpy()):
                print("\nfailed at index ", i)
                failed = True
    if not failed:
        print("-- passed")
