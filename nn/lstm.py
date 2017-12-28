import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .rnn_cell_base import RNNCellBase

class LSTM(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, sigmoid=None, tanh=None):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.sigmoid = self.get_activation(sigmoid)
        self.tanh = self.get_activation(tanh)
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
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
        self.hidden[0].detach_()
        self.hidden[1].detach_()
        self.weight_ih.detach_()
        self.weight_hh.detach_()
        self.bias_ih.detach_()
        self.bias_hh.detach_()

    def forward(self, input_data, future=0):
        timesteps, batch_size, features = input_data.size()
        # print("t %d, b %d, f %d" % (timesteps, batch_size, features))
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size), requires_grad=False)
        
        if self.hidden is None:
            self.hidden = (Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False), # h
                           Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)) # c
        
        self.check_forward_input(input_data[0])
        self.check_forward_hidden(input_data[0], self.hidden[0], '[0]')
        self.check_forward_hidden(input_data[0], self.hidden[1], '[1]')
        
        for i, input_t in enumerate(input_data.split(1)):

            hx, cx = self.hidden
            gates = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih) + \
                     F.linear(hx, self.weight_hh, self.bias_hh)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = self.sigmoid(ingate)
            forgetgate = self.sigmoid(forgetgate)
            cellgate = self.tanh(cellgate)
            outgate = self.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)
            
            self.hidden = hy, cy
            outputs[i] = self.hidden[0]
        
        return outputs

class TLSTM(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(TLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
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
        self.hidden[0].detach_()
        self.hidden[1].detach_()
        self.weight_ih.detach_()
        self.weight_hh.detach_()
        self.bias_ih.detach_()
        self.bias_hh.detach_()

    def forward(self, input_data, future=0):
        timesteps, batch_size, features = input_data.size()
        # print("t %d, b %d, f %d" % (timesteps, batch_size, features))
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size), requires_grad=False)
        
        if self.hidden is None:
            self.hidden = (Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False), # h
                           Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)) # c
        
        self.check_forward_input(input_data[0])
        self.check_forward_hidden(input_data[0], self.hidden[0], '[0]')
        self.check_forward_hidden(input_data[0], self.hidden[1], '[1]')
        
        for i, input_t in enumerate(input_data.split(1)):

            hx, cx = self.hidden
            gates = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih) + \
                     F.linear(hx, self.weight_hh, self.bias_hh)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)
            
            self.hidden = hy, cy
            outputs[i] = self.hidden[0]
        
        return outputs
