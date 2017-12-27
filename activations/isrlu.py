import torch
from torch.autograd import Function

class ISRLU(Function):
    @staticmethod
    def forward(ctx, tensor, alpha=1):
        negatives = torch.min(tensor, 0 * tensor)
        nisr = torch.rsqrt(1. + alpha * (negatives ** 2))
        # nisr == 1 where tensor elements are positive
        return tensor * nisr

class ISRU_tanh(Function):
    @staticmethod
    def forward(ctx, tensor, alpha=1):
        return tensor * torch.rsqrt(1. + alpha * (tensor ** 2))

class ISRU_sigmoid(Function):
    @staticmethod
    def forward(ctx, tensor, alpha=1):
        return .5 + .5 * tensor * torch.rsqrt(1. + alpha * (tensor ** 2))


if __name__ == "__main__":
    import numpy as np

    data = np.random.rand(10)
    alpha = 1.

    print("Testing ISRLU", end=" ")
    pos = data > 0
    calc = np.empty_like(data)
    calc[pos] = data[pos]
    calc[~pos] = 1. / np.sqrt(1. + alpha * (data[~pos] ** 2))
    out = ISRLU.forward(None, torch.Tensor(data), alpha).numpy()
    print("--", "passed" if np.allclose(calc, out) else "failed")

    print("Testing ISRU_tanh", end=" ")
    calc = data / np.sqrt(1. + alpha * (data ** 2))
    out = ISRU_tanh.forward(None, torch.Tensor(data), alpha).numpy()
    print("--", "passed" if np.allclose(calc, out) else "failed")

    print("Testing ISRU_sigmoid", end=" ")
    calc = .5 + .5 * data / np.sqrt(1. + alpha * (data ** 2))
    out = ISRU_sigmoid.forward(None, torch.Tensor(data), alpha).numpy()
    print("--", "passed" if np.allclose(calc, out) else "failed")

ISRLU        = ISRLU.apply
ISRU_tanh    = ISRU_tanh.apply
ISRU_sigmoid = ISRU_sigmoid.apply
