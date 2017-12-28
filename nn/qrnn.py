import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .rnn_cell_base import RNNCellBase

class fakeQRNN(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, sigmoid=None, tanh=None):
        super(fakeQRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.sigmoid = self.get_activation(sigmoid)
        self.tanh = self.get_activation(tanh)
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
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
        self.weight_ih.detach_()
        self.bias_ih.detach_()
    
    def detach_hidden(self):
        self.hidden.detach_()
        self.weight_ih.detach_()
        self.bias_ih.detach_()

    def forward(self, input_data, future=0):
        timesteps, batch_size, features = input_data.size()
        # print("t %d, b %d, f %d" % (timesteps, batch_size, features))
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size), requires_grad=False)
        
        if self.hidden is None:
            self.hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        
        self.check_forward_input(input_data[0])
        self.check_forward_hidden(input_data[0], self.hidden)
        
        for i, input_t in enumerate(input_data.split(1)):
            x = input_t.view(batch_size, features)
            gi = F.linear(x, self.weight_ih, self.bias_ih)
            i_r, i_f, i_n = gi.chunk(3, 1)

            readgate = self.sigmoid(i_r)
            forgetgate = self.sigmoid(i_f)
            newgate = self.tanh(i_n)
            self.hidden = newgate + forgetgate * (self.hidden - newgate)
            outputs[i] = readgate * self.hidden
        
        return outputs
