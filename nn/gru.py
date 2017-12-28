import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .rnn_cell_base import RNNCellBase

class GRU(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, sigmoid=None, tanh=None):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.sigmoid = F.sigmoid if sigmoid is None else self.get_activation(sigmoid)
        self.tanh = F.tanh if tanh is None else self.get_activation(tanh)
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.hidden = None
    
    def reset_hidden(self):
        self.hidden = None
        self.weight_ih.detach_()
        self.weight_hh.detach_()
        self.bias_ih.detach_()
        self.bias_hh.detach_()
    
    def detach_hidden(self):
        self.hidden.detach_()
        self.weight_ih.detach_()
        self.weight_hh.detach_()
        self.bias_ih.detach_()
        self.bias_hh.detach_()

    def forward(self, input_data, future=0):
        timesteps, batch_size, features = input_data.size()
        # print("t %d, b %d, f %d" % (timesteps, batch_size, features))
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size), requires_grad=False)
        
        if self.hidden is None:
            self.hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        
        self.check_forward_input(input_data[0])
        self.check_forward_hidden(input_data[0], self.hidden)
        
        for i, input_t in enumerate(input_data.split(1)):
            
            gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih)
            gh = F.linear(self.hidden, self.weight_hh, self.bias_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = self.sigmoid(i_r + h_r)
            inputgate = self.sigmoid(i_i + h_i)
            newgate = self.tanh(i_n + resetgate * h_n)
            self.hidden = newgate + inputgate * (self.hidden - newgate)
            outputs[i] = self.hidden
            """
            outputs[i] = self._backend.GRUCell(
                input_t.view(batch_size, features), self.hidden,
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )
            #"""
        
        return outputs
