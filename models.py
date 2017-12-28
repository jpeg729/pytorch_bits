
from torch import nn
import nn as custom


class Model(nn.Module):
    def __init__(self, input_size=1, layers=["LSTM_51"], output_size=1, sigmoid=None, tanh=None):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        prev_size = input_size
        for l, spec in enumerate(layers):
            bits = spec.split("_")
            cell_type = bits[0]
            hidden_size = int(bits[1])
            print("Adding layer of type", spec, ":", prev_size, "->", hidden_size, *bits[2:])
            layer = getattr(custom, cell_type)(
                input_size=prev_size, hidden_size=hidden_size, *bits[2:],
                sigmoid=sigmoid, tanh=tanh)
            self.layers.append(layer)
            self.add_module("layer"+str(l), layer)
            prev_size = hidden_size
        if prev_size != output_size:
            print("Adding linear layer :", prev_size, "->", output_size)
            layer = nn.Linear(prev_size, output_size)
            self.layers.append(layer)
            self.add_module("layer"+str(l+1), layer)

    def reset_hidden(self):
        for layer in self.layers:
            if hasattr(layer, "reset_hidden"):
                layer.reset_hidden()
        # for module in self.modules():
        #     if module is not self and hasattr(module, "reset_hidden"):
        #         module.reset_hidden()
    
    def detach_hidden(self):
        for layer in self.layers:
            if hasattr(layer, "detach_hidden"):
                layer.detach_hidden()

    def forward(self, data, future=0):
        for layer in self.layers:
            data = layer(data)
        return data
