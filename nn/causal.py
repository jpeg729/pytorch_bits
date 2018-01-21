import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 sigmoid=None,
                 tanh=None):
        self.left_padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            input_size,
            hidden_size,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        # data is in shape (timesteps, batches, features)
        # conv needs shape (batches, features, timesteps)
        x = F.pad(input.permute(1, 2, 0), (self.left_padding,0))
        conv_out = super(CausalConv1d, self).forward(x)
        # must return shape (timesteps, batches, features)
        return conv_out.permute(2, 0, 1)

class Wave(nn.Module):
    def __init__(self, input_size, hidden_size, layers=3, activation="tanh"):
        super(Wave, self).__init__()
        self.layers = []
        prev_size = input_size
        for layer in range(layers):
            conv = CausalConv1d(prev_size, hidden_size, kernel_size=2, dilation=2**layer)
            self.layers.append(conv)
            self.add_module("layer"+str(layer), conv)
            prev_size = hidden_size

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data

class ShortWave(nn.Module):
    def __init__(self, input_size, hidden_size, layers=3):
        super(ShortWave, self).__init__()
        self.layers = []
        prev_size = input_size
        for layer in range(layers):
            conv = CausalConv1d(prev_size, hidden_size, kernel_size=2, dilation=1)
            self.layers.append(conv)
            self.add_module("layer"+str(layer), conv)
            prev_size = hidden_size

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data

def test_CausalConv1d(timesteps, input_size, hidden_size, batch_size, kernel_size, dilation, bias):
    m = CausalConv1d(input_size, hidden_size, kernel_size=kernel_size, dilation=dilation, bias=bias!=0)
    m.weight.data.fill_(1)
    if bias:
        m.bias.data.fill_(bias)
    x = torch.autograd.Variable(torch.zeros(timesteps, batch_size, input_size), requires_grad=False)
    
    for batch in range(batch_size):
        for t in range(timesteps):
            for ci in range(input_size):
                x.data.fill_(0)
                x[t, batch, ci] = 1
                out = m(x)
                for b in range(batch_size):
                    for co in range(hidden_size):
                        if b == batch:
                            target = [1+bias if j in range(t, t+k*d, d) else bias for j in range(timesteps)]
                        else:
                            target = [bias for j in range(timesteps)]
                        if list(out[:, b, co].data) != target:
                            print("\nCausalConv1d wrong output for kernel_size", k, 
                                "and dilation", d, "i", input_size, "out", hidden_size,
                                "batch_size", batch_size, 
                                "bias", bias)
                            print("input ", " ".join(str(int(el)) for el in x[:, b, co].data))
                            print("output", " ".join(str(el) for el in out[:, b, co].data))
                            print("target", " ".join(str(el) for el in target))
                            assert list(out[:, b, co].data) == target, "Test failed"

if __name__ == "__main__":
    import numpy as np
    timesteps, batch_size = 20, 3
    print("Running tests", end="")
    for ci in range(1, 3):
        for co in range(1, 3):
            for k in range(1, 4):
                for d in range(1, 3):
                    print(".", end="", flush=True)
                    test_CausalConv1d(timesteps, ci, co, batch_size, k, d, 0.5)
                    test_CausalConv1d(timesteps, ci, co, batch_size, k, d, 0)
    print("\nCausalConv1d tests passed")
