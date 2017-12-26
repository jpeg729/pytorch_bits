import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .rnn_cell_base import RNNCellBase

class SRUf(RNNCellBase):
    """The simplest SRU mentioned in the paper."""

    def __init__(self, input_size, hidden_size, bias=True, sigmoid=F.sigmoid, tanh=F.tanh):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.sigmoid = sigmoid
        self.tanh = tanh
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
            
            gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih)
            i_f, i_n = gi.chunk(2, 1)

            forgetgate = self.sigmoid(i_f)
            newgate = i_n
            self.hidden = newgate + forgetgate * (self.hidden - newgate)
            outputs[i] = self.tanh(self.hidden)
        
        return outputs

class SRU2(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(SRU2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
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
            
            gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih)
            i_i, i_f, i_n = gi.chunk(3, 1)

            inputgate = F.sigmoid(i_i)
            forgetgate = F.sigmoid(i_f)
            newgate = i_n
            self.hidden = inputgate * newgate + forgetgate * self.hidden
            outputs[i] = F.tanh(self.hidden)
        
        return outputs

class SRU(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
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
        # """
        gis = F.linear(input_data, self.weight_ih, self.bias_ih)
        
        for i, gi in enumerate(gis.split(1)):
            i_r, i_f, i_n = gi.squeeze().chunk(3, 1)

            readgate = F.sigmoid(i_r)
            forgetgate = F.sigmoid(i_f)
            newgate = i_n
            self.hidden = newgate + forgetgate * (self.hidden - newgate)
            outputs[i] = newgate + readgate * (F.tanh(self.hidden) - newgate)
        """
        for i, input_t in enumerate(input_data.split(1)):
            x = input_t.view(batch_size, features)
            gi = F.linear(x, self.weight_ih, self.bias_ih)
            i_r, i_f, i_n = gi.chunk(3, 1)

            readgate = F.sigmoid(i_r)
            forgetgate = F.sigmoid(i_f)
            newgate = i_n
            self.hidden = newgate + forgetgate * (self.hidden - newgate)
            outputs[i] = newgate + readgate * (F.tanh(self.hidden) - newgate)
        #"""
        return outputs
