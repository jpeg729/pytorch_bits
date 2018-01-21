import torch
import torch.nn.functional as F
from torch.autograd import Function

# f(x) = 1.5 * tanh(x) + 0.5 * tanh(−3 * x)

class TernaryTanh(Function):
    @staticmethod
    def forward(ctx, tensor):
        return 1.5 * F.tanh(tensor) + 0.5 * F.tanh(-3 * tensor)

TernaryTanh = TernaryTanh.apply