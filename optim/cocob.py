import torch
from torch.optim.optimizer import Optimizer, required

class COCOB(Optimizer):

    def __init__(self, params, alpha=100, weight_decay=False):
        defaults = dict(alpha=alpha, weight_decay=weight_decay)
        super(COCOB, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(COCOB, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)

                state = self.state[p]
                if len(state) == 0:
                    state['L'] = torch.zeros_like(p.data)
                    state['gradients_sum'] = torch.zeros_like(p.data)
                    state['grad_norm_sum'] = torch.zeros_like(p.data)
                    state['reward'] = torch.zeros_like(p.data)
                    state['w'] = torch.zeros_like(p.data)

                L = state['L']
                reward = state['reward']
                gradients_sum = state['gradients_sum']
                grad_norm_sum = state['grad_norm_sum']
                old_w = state['w']

                torch.max(L, torch.abs(d_p), out=L)
                torch.max(reward - old_w * d_p, torch.Tensor([0]), out=reward)
                gradients_sum.add_(d_p)
                grad_norm_sum.add_(torch.abs(d_p))

                # the paper sets weights_t = weights_1 + new_w
                # we use the equivalent formula: weights_t = weights_tm1 - old_w + new_w
                new_w = state['w'] = -gradients_sum / (L * torch.max(grad_norm_sum + L, alpha * L)) * (L + reward)
                p.data.add_(-1, old_w)
                p.data.add_(new_w)

        return loss