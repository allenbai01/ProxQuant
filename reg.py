import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from bst import *
import pdb

# Defines what parameters to add binary reg
def if_binary(n):
    return (not('bn' in n) and not('downsample' in n)
            and not('fc' in n and 'bias' in n))
def if_binary_tern(n):
    return (not('bn' in n) and not('downsample' in n)
            and not('fc' in n) and not(n == 'conv1.weight'))
def if_binary_lm(n):
    return ('weight' in n)
    # return ('weight' in n) and ('encoder' not in n)
    # return ('weight' in n) and ('coder' not in n)

def binary_reg(net):
    '''Binary-enforcing regularization for deep nets'''
    return sum([torch.min(torch.abs(w-1), torch.abs(w+1)).mean()
                for n, w in net.named_parameters() if if_binary(n)])

def adjust_bn(model):
    '''Turn off running stats tracking in BN layers'''
    for n, m in model.named_modules():
        if 'bn' in n:
            m.track_running_stats = False
    
def stochastic_binarize(A):
    '''Stochastic binarization of a Float tensor'''
    return (torch.rand_like(A) < ((A + 1) / 2)).mul(2).float() - 1
def exponential_binarize(A):
    '''Exponential binarization of a Float tensor'''
    A_exp = A.exp() / (A.exp() + (-A).exp())
    return (torch.rand_like(A) < A_exp).mul(2).float() - 1

def ternarize(A, delta_method='mean'):
    '''Ternarize a Float tensor'''
    A_quant = A.clone()
    # inds_one, inds_zero, inds_mone = (A >= 0.5), (A.abs() < 0.5), (A <= -0.5)
    # A_quant.masked_fill_(inds_one, 1.0)
    # A_quant.masked_fill_(inds_zero, 0.0)
    # A_quant.masked_fill_(inds_mone, -1.0)
    if delta_method == 'max':
        delta = A.abs().max() * 0.05
    elif delta_method == 'mean':
        delta = A.abs().mean() * 0.7
    A_quant.masked_fill_(A.abs() < delta, 0)
    inds_p, inds_n = (A >= delta), (A <= -delta)
    A_quant.masked_fill_(inds_p, A[inds_p].mean())
    A_quant.masked_fill_(inds_n, A[inds_n].mean())
    return A_quant
def greedy_median(w, n_bits=1, by_row=False):
    '''Greedy median quantization for tensors'''
    b_list, alpha_list = [], []
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        b = r.sign()
        # Break sign ties randomly
        b[b == 0] = torch.randn_like(b[b == 0]).sign()
        if by_row:
            alpha, _ = r.abs().median(dim=1, keepdim=True)
        else:
            alpha = r.abs().median()
        r -= b * alpha
        w_hat += b * alpha
        b_list += [b]
        alpha_list += [alpha]
    return w_hat, b_list, alpha_list
def soft_threshold(w, w_hat, reg=0.0):
    '''Soft threshold a tensor towards another tensor'''
    w_sign, w_res = (w-w_hat).sign(), (w-w_hat).abs()
    return w_hat + w_sign * F.relu(w_res - reg)

def greedy_mean(w, n_bits=1, by_row=False):
    '''Greedy mean quantization for tensors'''
    B = torch.zeros(w.shape + (n_bits,), device=w.device)
    if by_row:
        Alpha = torch.zeros(w.shape[0], n_bits, device=w.device)
    else:
        Alpha = torch.zeros(n_bits, device=w.device)
    # b_list, alpha_list = [], []
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        b = r.sign()
        # Break sign ties randomly
        b[b == 0] = torch.randn_like(b[b == 0]).sign()
        if by_row:
            alpha = r.abs().mean(dim=1, keepdim=True)
        else:
            alpha = r.abs().mean()
        r -= b * alpha
        w_hat += b * alpha
        B[:,:,i] = b
        if by_row:
            Alpha[:,i] = alpha.view(-1)
        else:
            Alpha[i] = alpha
        # b_list += [b]
        # alpha_list += [alpha]
    return w_hat, B, Alpha
def batch_inverse(B):
    '''Batch-inverse 2x2 matrices in a stacked nx2x2 tensor'''
    dets = B[:,0,0] * B[:,1,1] - B[:,0,1] * B[:,1,0]
    B_inv = torch.zeros_like(B)
    B_inv[:,0,0] = B[:,1,1]
    B_inv[:,1,1] = B[:,0,0]
    B_inv[:,0,1] = -B[:,0,1]
    B_inv[:,1,0] = -B[:,1,0]
    B_inv /= dets.view(-1, 1, 1)
    return B_inv
def batch_cg(A, b, x=None, n_steps=None, eps=1e-6):
    '''
    Batch conjugate gradient for solving Ax = b
    A is d1xkxk, b is d1xk, x is d1xk
    '''
    d1, k, _ = A.shape
    if n_steps == None:
        n_steps = k
    # if x == None:
    #     x = torch.zeros_like(b)
    # Initialize
    x = x.clone().view(d1, k, 1)
    b = b.view(d1, k, 1)
    r = b - A.bmm(x)
    rtr_new = r.transpose(1, 2).bmm(r)
    p = r.clone()
    # Perform batch CG
    for i in range(n_steps):
        rtr = rtr_new
        Ap = A.bmm(p)
        alpha = rtr / (p.transpose(1, 2).bmm(Ap) + eps)
        x += alpha * p
        r -= alpha * Ap
        rtr_new = r.transpose(1, 2).bmm(r)
        beta = rtr_new / (rtr + eps)
        p = r + beta * p
    return x.view(d1, k)
def refine_mean(w, w_hat, B, Alpha, by_row=False,
                eps_inv=1e-6, cond_thres=20):
    '''Refine a (mean) quantization'''
    d1, d2 = w.shape
    eta = 1 / (2 * d2)
    least_squares = False
    with torch.no_grad():
        n_bits = B.shape[-1]
        r, w_hat_new = w.clone(), 0.
        if by_row:
            B_cov = B.transpose(1, 2).bmm(B)
            Btw = B.transpose(1, 2).bmm(w.view(d1, d2, 1)).view(d1, n_bits)
            if least_squares:
                # Find Alpha_new via row-wise least squares (temp. disabled)
                B_cov_numpy = B_cov.cpu().detach().numpy()
                B_cov_inv = torch.from_numpy(np.linalg.inv(
                    B_cov_numpy + \
                    np.tile(np.eye(n_bits, dtype=np.float32), (B.shape[0], 1, 1)) \
                    * eps_inv)).to(B_cov.device)
                B_reg = torch.einsum('ijk,ilk->ilj', (B_cov_inv, B))
                Alpha_new = torch.einsum('ijk,ij->ik', (B_reg, w))
                # # Condition number thresholding
                # bad_inds = np.where(np.linalg.cond(B_cov_numpy) > cond_thres)[0]
                # Alpha_new[bad_inds,:] = Alpha[bad_inds,:]
            else:
                # Find Alpha_new via conjugate gradients
                Alpha_new = batch_cg(B_cov, Btw, x=Alpha)
            Alpha_new, _ = Alpha_new.abs().sort(descending=True)
            # Find B_new via greedy or binary search.
            # Closed-form solution for <=2 bits, BST for >=3 bits
            if n_bits <= 3:
                B_new = torch.zeros_like(B)
                for i in range(n_bits):
                    B_new[:,:,i] = r.sign()
                    r -= B_new[:,:,i] * Alpha_new[:,i].view([-1, 1])
                    w_hat_new += B_new[:,:,i] * Alpha_new[:,i].view([-1, 1])
            else:
                B_new = find_B_torch(w, Alpha_new, by_row=True)
                w_hat_new = torch.einsum('ijl,il->ij', (B_new, Alpha_new))
        else:
            w_flat = w.view(-1)
            B_flat = B.view(-1, n_bits)
            # Find Alpha_new via least squares
            Alpha_new = B_flat.t().mm(B_flat).inverse().mm(B_flat.t()).mv(w_flat)
            Alpha_new, _ = Alpha_new.abs().sort(descending=True)
            # Find B_new via greedy or binary search.
            if n_bits <= 4:
                B_new = torch.zeros_like(B)
                for i in range(n_bits):
                    B_new[:,:,i] = r.sign()
                    r -= B_new[:,:,i] * Alpha_new[i]
                    w_hat_new += B_new[:,:,i] * Alpha_new[i]
            else:
                B_new = find_B_torch(w, Alpha_new, by_row=False)
                w_hat_new = B_new.mv(Alpha_new)
    return w_hat_new, B_new, Alpha_new

def alt_quantize(w, n_bits=1, by_row=False, n_rounds=1,
                 norm_rate=1.0):
    '''Alternating multi-bit quantization'''
    w_hat, B, Alpha = greedy_mean(w, n_bits=n_bits, by_row=by_row)
    n_rounds -= 1
    if n_bits > 1:
        while n_rounds > 0:
            w_hat, B, Alpha = refine_mean(w, w_hat, B, Alpha, by_row=by_row)
            n_rounds -= 1
    w_hat.mul_(norm_rate)
    return w_hat, B, Alpha
class AltQuantize(Function):
    '''Alternating quantization for auto-grad'''
    def __init__(self, n_bits=1, by_row=False, n_rounds=3):
        super(AltQuantize, self).__init__()
        self.n_bits = n_bits
        self.by_row = by_row
        self.n_rounds = n_rounds
        
    def forward(self, input):
        # Note: need to reshape 1 x bs x emb_dim into bs x emb_dim
        if isinstance(input, tuple):
            input_squeezed, input_shape = input[0].squeeze(), input[0].shape
            input_quantized, _, _ = alt_quantize(
                input_squeezed, n_bits=self.n_bits, by_row=self.by_row, n_rounds=self.n_rounds)
            return (input_quantized.view(input_shape),) + input[1:]
        else:
            input_squeezed, input_shape = input.squeeze(), input.shape
            input_quantized, _, _ = alt_quantize(
                input_squeezed, n_bits=self.n_bits, by_row=self.by_row, n_rounds=self.n_rounds)
            return input_quantized.view(input_shape)
                
    def backward(self, grad_output):
        '''Straight-through back propagation'''
        # return grad_output.clone()
        return grad_output

class BinOp():
    '''Class for quantization operations on nn.Module'''
    def __init__(self, model, if_binary=if_binary,
                 ttq=False):
        self.model = model
        # Make copy of all parameters and store in saved_params
        self.saved_params = {}
        self.init_params = {}
        self.if_binary = if_binary
        for n, p in model.named_parameters():
            if self.if_binary(n):
                self.saved_params[n] = p.data.clone()
                self.init_params[n] = p.data.clone()
        self.ttq = ttq
        if ttq:
            self.ternary_assigns = {n: ([], [])
                                    for n, p in model.named_parameters()
                                    if self.if_binary(n)}
            self.ternary_vals = {}
            # self.ternary_vals.update({n + "_pos": p.data[p.data >= 0].mean().detach()
            #                           for n, p in model.named_parameters()
            #                           if self.if_binary(n)})
            # self.ternary_vals.update({n + "_neg": p.data[p.data < 0].mean().detach()
            #                           for n, p in model.named_parameters()
            #                           if self.if_binary(n)})
            self.ternary_vals.update({n + "_pos": torch.ones([1]).mean().to(p.device)
                                      for n, p in model.named_parameters()
                                      if self.if_binary(n)})
            self.ternary_vals.update({n + "_neg": torch.ones([1]).mean().mul(-1).to(p.device)
                                      for n, p in model.named_parameters()
                                      if self.if_binary(n)})

    def prox_operator(self, reg, reg_type='binary',
                      n_bits=1, by_row=False, n_rounds=1,
                      norm_rate=1.0):
        '''
        Compute prox operator and overwrite self.model
        Note: when reg >= 1 and clip is on (weights are in [-1, 1]),
              binary prox is equivalent to sign.
        '''
        if reg_type == 'binary':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_sign, p_abs = p.data.sign(), p.data.abs()
                    p.data.copy_(p_sign * (F.relu((p_abs - 1).abs() - reg) \
                                           * (p_abs - 1).sign() + 1))
        if reg_type == 'ternary':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_((p.data + ternarize(p.data) * reg) / (1 + reg))
        elif reg_type == 'median':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_quant, _, _ = greedy_median(p.data, n_bits=n_bits, by_row=by_row)
                    p.data.copy_(soft_threshold(p, p_quant, reg))
                    # p_sign, p_abs = p.data.sign(), p.data.abs()
                    # if by_row:
                    #     p_med, _ = p_abs.median(dim=1, keepdim=True)
                    # else:
                    #     p_med = p_abs.median()
                    # p.data.copy_(p_sign * (F.relu((p_abs - p_med).abs() - reg) \
                    #                        * (p_abs - p_med).sign() + p_med))
        elif reg_type == 'mean':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_prox = p.data.clone()
                    for _ in range(n_rounds):
                        p_quant, _, _ = alt_quantize(p_prox, n_bits=n_bits,
                                                     by_row=by_row, n_rounds=3,
                                                     norm_rate=norm_rate)
                        p_prox = (p.data + p_quant * reg) / (1 + reg)
                    p.data.copy_(p_prox)
                    # p.data.copy_(soft_threshold(p, p_quant, reg))
    def quantize(self, mode='deterministic',
                 n_bits=1, by_row=False, n_rounds=1,
                 norm_rate=1.0):
        '''Quantize the model'''
        if mode == 'deterministic':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(p.data.sign())
        elif mode == 'binary_freeze':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(p.data.sign())
                    p.requires_grad = False
        elif mode == 'ternary':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(ternarize(p))
        elif mode == 'ternary_freeze':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(ternarize(p))
                    p.requires_grad = False
        elif mode == 'ttq':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    delta = p.data.abs().max() * 0.05
                    inds_p, inds_n = p.data >= delta, p.data <= -delta
                    self.ternary_assigns[n] = inds_p, inds_n
                    p.data.masked_fill_(inds_p, self.ternary_vals[n + "_pos"])
                    p.data.masked_fill_(inds_n, self.ternary_vals[n + "_neg"])
        elif mode == 'median':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_quant, _, _ = greedy_median(p.data, n_bits=n_bits, by_row=by_row)
                    p.data.copy_(p_quant)
        elif mode == 'mean':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_quant, _, _ = alt_quantize(p.data, n_bits=n_bits,
                                                 by_row=by_row, n_rounds=n_rounds,
                                                 norm_rate=norm_rate)
                    p.data.copy_(p_quant)
        elif mode == 'mean_freeze':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p_quant, _, _ = alt_quantize(p.data, n_bits=n_bits,
                                                 by_row=by_row, n_rounds=n_rounds)
                    p.data.copy_(p_quant)
                    p.requires_grad = False
        elif mode == 'stochastic':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(stochastic_binarize(p.data))
        elif mode == 'exponential':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.copy_(exponential_binarize(p.data))

    def quantize_error(self, mode='deterministic',
                       n_bits=1, by_row=False, n_rounds=1):
        '''Compute quantization error'''
        self.save_params()
        self.quantize(mode=mode, n_bits=n_bits, by_row=by_row, n_rounds=n_rounds)
        results = {'quant_error_' + n: (p.data - self.saved_params[n]).norm() / self.saved_params[n].norm()
                   for n, p in self.model.named_parameters() if self.if_binary(n)}
        self.restore()
        return results
    
    def clip(self, clip_type='binary'):
        '''Clip the parameters'''
        if clip_type == 'binary':
            for n, p in self.model.named_parameters():
                if self.if_binary(n):
                    p.data.clamp_(-1, 1)
    def wdecay(self, wd):
        '''Perform weight decay on quantized parameters'''
        for n, p in self.model.named_parameters():
            if self.if_binary(n):
                p.data.mul_(1 - wd)
    def save_params(self):
        for n, p in self.model.named_parameters():
            if self.if_binary(n):
                self.saved_params[n].copy_(p.data)
    def restore(self):
        for n, p in self.model.named_parameters():
            if self.if_binary(n):
                p.data.copy_(self.saved_params[n])
                if self.ttq:
                    # Modify gradients according to ternary vals
                    inds_p, inds_n = self.ternary_assigns[n]
                    p.grad[inds_p].mul_(self.ternary_vals[n + "_pos"].abs())
                    p.grad[inds_n].mul_(self.ternary_vals[n + "_neg"].abs())
        
def binary_levels(net, if_binary=if_binary):
    '''Compute level of binarization'''
    return {'binary_' + n: torch.min(torch.abs(w-1), torch.abs(w+1)).mean()
            for n, w in net.named_parameters() if if_binary(n)}
def median_levels(net, if_binary=if_binary,
                  n_bits=1, by_row=False):
    results = {}
    for n, p in net.named_parameters():
        if if_binary(n):
            p_quant, b_list, alpha_list = greedy_median(p.data, n_bits=n_bits, by_row=by_row)
            results['median_' + n] = alpha_list[0].mean()
            results['medness_' + n] = (p.data - p_quant).abs().mean() / alpha_list[0].mean()
    return results
def alt_quantization_error(net, if_binary=if_binary,
                           n_bits=1, by_row=False, n_rounds=1):
    results = {}
    for n, p in net.named_parameters():
        if if_binary(n):
            p_quant, _, _ = alt_quantize(p.data,
                                         n_bits=n_bits, by_row=by_row, n_rounds=n_rounds)
            results['alt_quantize_error_' + n] = (p.data - p_quant).norm() / p.data.norm()
    return results
def sign_changes(bin_op):
    '''Compute fraction of sign changes between curr and initialization'''
    return {'sign_change_' + n:
            (p.sign() != bin_op.init_params[n].sign()).sum().item() / p.numel()
            for n, p in bin_op.model.named_parameters() if bin_op.if_binary(n)}
    
def adjust_reg(reg, epoch, total_epochs, max_reg=1.0):
    """Reconfigures the regularization strength"""
    return min(reg * epoch / total_epochs, max_reg)

def step_ternary_vals(bin_op, optimizer):
    """Perform gradient step on the ternary values"""
    assert(bin_op.ttq)
    curr_lr = optimizer.param_groups[0]['lr']
    for n, p in bin_op.model.named_parameters():
        if bin_op.if_binary(n):
            grad_pos = p.grad[bin_op.ternary_assigns[n][0]].sum()
            grad_neg = p.grad[bin_op.ternary_assigns[n][1]].sum()
            # if n == 'conv1.weight':
            #     pdb.set_trace()
            bin_op.ternary_vals[n + "_pos"].add_(-curr_lr * grad_pos)
            bin_op.ternary_vals[n + "_neg"].add_(-curr_lr * grad_neg)
