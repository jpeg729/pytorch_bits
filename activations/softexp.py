import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

testing = False

class SoftExp(nn.Module):
    def __init__(self, input_size):
        super(SoftExp, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(input_size))

    def forward(self, data):
        self.alpha.data.clamp_(-1, 1)
        
        positives = torch.gt(F.threshold(self.alpha, 0, 0), 0)
        negatives = torch.gt(F.threshold(-self.alpha, 0, 0), 0)

        output = data.clone()
        pos_out = (torch.exp(self.alpha * data) - 1) / self.alpha + self.alpha
        neg_out = -(torch.log(1 - self.alpha * (data + self.alpha))) / self.alpha
        
        output.masked_scatter_(positives, pos_out.masked_select(positives))
        output.masked_scatter_(negatives, neg_out.masked_select(negatives))
        return output

if __name__ == "__main__":
    import numpy as np

    testing = True
    test_size = 50
    data = Variable(torch.abs(torch.randn(1, 1, test_size)))
    softexp = SoftExp(test_size)
    softexp.alpha.data.copy_(torch.randn(test_size))
    softexp.alpha.data[0] = 0


    print("Testing SoftExp", end=" ")
    out = softexp(data)
    loss = nn.MSELoss()(out, torch.ones_like(data))
    loss.backward()

    failed = False
    for i in range(test_size):
        if softexp.alpha.data[i] == 0:
            activation = data
            if not np.allclose(out[:, :, i].data.numpy(), activation[:, :, i].data.numpy()):
                print("\nfailed at index", i, "for zero alpha", softexp.alpha.data[i], "output", out[:, :, i].data.numpy(), "for data", data[:, :, i].data.numpy(), end="")
                failed = True
        elif softexp.alpha.data[i] > 0:
            activation = ((torch.exp(softexp.alpha * data) - 1) / softexp.alpha + softexp.alpha)
            if not np.allclose(out[:, :, i].data.numpy(), activation[:, :, i].data.numpy()):
                print("\nfailed at index", i, "for positive alpha", softexp.alpha.data[i], "output", out[:, :, i].data.numpy(), "for data", data[:, :, i].data.numpy(), end="")
                failed = True
        elif softexp.alpha.data[i] < 0:
            activation = (-(torch.log(1 - softexp.alpha * (data + softexp.alpha))) / softexp.alpha)
            if not np.allclose(out[:, :, i].data.numpy(), activation[:, :, i].data.numpy()):
                print("\nfailed at index", i, "for negative alpha", softexp.alpha.data[i], "output", out[:, :, i].data.numpy(), "for data", data[:, :, i].data.numpy(), end="")
                failed = True
    print("-- passed" if not failed else "")
