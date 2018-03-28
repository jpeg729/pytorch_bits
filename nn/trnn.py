import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .rnn_cell_base import RNNCellBase

class TRNN(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, sigmoid=None, tanh=None):
        super(TRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.sigmoid = F.sigmoid if sigmoid is None else self.get_activation(sigmoid)
        self.tanh = F.tanh if tanh is None else self.get_activation(tanh)
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(2 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.hidden = None
    
    def reset_hidden(self):
        self.hidden = None
    
    def detach_hidden(self):
        self.hidden.detach_()

    def forward(self, input_data, future=0):
        timesteps, batch_size, features = input_data.size()
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size), requires_grad=False)
        
        if self.hidden is None:
            self.hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        
        self.check_forward_input(input_data[0])
        self.check_forward_hidden(input_data[0], self.hidden)
        
        for i, input_t in enumerate(input_data.split(1)):
            
            gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih)
            i_n, i_f = gi.chunk(2, 1)

            forgetgate = self.sigmoid(i_f)
            newgate = i_n
            self.hidden = newgate + forgetgate * (self.hidden - newgate)
            outputs[i] = self.hidden
        
        return outputs
