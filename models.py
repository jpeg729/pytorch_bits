
from torch import nn
import nn as custom


class Model(nn.Module):
    def __init__(self, input_size=1, layers=["LSTM_51"], output_size=1, sigmoid=None, tanh=None, biases=True):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        prev_size = input_size
        for l, spec in enumerate(layers):
            bits = spec.split("_")
            cell_type = bits.pop(0)
            print(spec, cell_type, bits)

            if hasattr(custom, cell_type):
                layer = getattr(custom, cell_type)
            elif hasattr(nn, cell_type):
                layer = getattr(nn, cell_type)

            layer_args = {}
            if "input_size" in layer.__init__.__code__.co_varnames:
                layer_args["input_size"] = prev_size
            if "hidden_size" in layer.__init__.__code__.co_varnames:
                layer_args["hidden_size"] = int(bits.pop(0))
                prev_size = layer_args["hidden_size"]
            
            for a in bits:
                print(a)
                k, v = a.split("=")
                k = k.replace("-", "_")
                if k not in layer.__init__.__code__.co_varnames:
                    print("kwarg", k, "for", cell_type, "not recognised")
                    continue
                for t in (int, float):
                    try:
                        v = t(v)
                        break
                    except ValueError:
                        pass
                layer_args[k] = v

            if "tanh" in layer.__init__.__code__.co_varnames:
                layer_args["tanh"] = tanh
            if "sigmoid" in layer.__init__.__code__.co_varnames:
                layer_args["sigmoid"] = sigmoid
            if "bias" in layer.__init__.__code__.co_varnames:
                layer_args["bias"] = biases

            print("Adding layer of type", spec, ":", layer_args)
            layer = layer(**layer_args,)
            self.layers.append(layer)
            self.add_module("layer"+str(l), layer)

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
