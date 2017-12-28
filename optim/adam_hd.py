import torch
from torch.optim.optimizer import Optimizer, required

class Adam_HD_lr_per_param(Optimizer):

    def __init__(self, params, lr=1e-3, lr_lr=.1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, lr_lr=lr_lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam_HD_lr_per_param, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam_HD does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['lr'] = torch.zeros_like(p.data).fill_(group['lr'])
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)
                    # For calculating df/dlr
                    state['m_debiased_tm1'] = torch.zeros_like(p.data)
                    state['v_debiased_tm1'] = torch.zeros_like(p.data)

                m, m_debiased_tm1 = state['m'], state['m_debiased_tm1']
                v, v_debiased_tm1 = state['v'], state['v_debiased_tm1']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias corrections
                m_debiased = m.div(1 - beta1 ** state['step'])
                v_debiased = v.div(1 - beta2 ** state['step'])

                # Update learning rate
                h = grad * (-m_debiased_tm1 / (torch.sqrt(v_debiased_tm1) + group['eps']))
                state['lr'].add_(-group['lr_lr'], h)

                p.data.addcdiv_(-state['lr'] * m_debiased, (torch.sqrt(v_debiased) + group['eps']))

                m_debiased_tm1.copy_(m_debiased)
                v_debiased_tm1.copy_(v_debiased)

        return loss

class Adam_HD(Optimizer):

    def __init__(self, params, lr=1e-3, lr_lr=.1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, lr_lr=lr_lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam_HD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam_HD does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['lr'] = group['lr']
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)
                    # For calculating df/dlr
                    state['m_debiased_tm1'] = torch.zeros_like(p.data)
                    state['v_debiased_tm1'] = torch.zeros_like(p.data)

                m, m_debiased_tm1 = state['m'], state['m_debiased_tm1']
                v, v_debiased_tm1 = state['v'], state['v_debiased_tm1']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias corrections
                m_debiased = m.div(1 - beta1 ** state['step'])
                v_debiased = v.div(1 - beta2 ** state['step'])

                # Update learning rate
                h = grad * (-m_debiased_tm1 / (torch.sqrt(v_debiased_tm1) + group['eps']))
                state['lr'] -= group['lr_lr'] * h.mean()

                p.data.addcdiv_(-state['lr'] * m_debiased, (torch.sqrt(v_debiased) + group['eps']))

                m_debiased_tm1.copy_(m_debiased)
                v_debiased_tm1.copy_(v_debiased)

        return loss