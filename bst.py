# Binary search tree
import numpy as np
import torch
def list_binary_vecs(n):
    ListBinaryVecs = {}
    for m in range(n+1):
        if m == 0:
            ListBinaryVecs[0] = [[]]
        else:
            ListBinaryVecs[m] = [[1.] + l for l in ListBinaryVecs[m-1]] \
                                + [[-1.] + l for l in ListBinaryVecs[m-1]]
    return ListBinaryVecs

ListBinaryVecs = list_binary_vecs(4)
BinaryMatrices = {n: np.vstack(ListBinaryVecs[n]).astype(np.float32)
                  for n in ListBinaryVecs}

def find_B_torch(w, Alpha, by_row=False):
    '''Find optimal quantization assignment via binary search'''
    n_bits = Alpha.shape[-1]
    bin_mat = torch.from_numpy(BinaryMatrices[n_bits]).to(w.device)
    if by_row:
        d1, d2 = w.shape
        row_inds = torch.arange(
            d1, dtype=torch.long).view(d1, 1).repeat([1, d2]).view(-1)
        # w is d1xd2, Alpha is d1xk, v is d1x2^k
        v = Alpha.mm(bin_mat.t())
        v_sorted, inds = torch.sort(v)
        # Binary search to find nearest neighbor
        w_flat = w.view([-1])
        Left = torch.zeros(d1*d2, dtype=torch.long, device=w.device)
        Right = torch.ones(d1*d2, dtype=torch.long, device=w.device) \
                * (2 ** n_bits - 1)
        for i in range(n_bits):
            Mid_Left = (Left + Right - 1) / 2
            Mid_Right = Mid_Left + 1
            mid_vals = (v_sorted[row_inds, Mid_Left] + \
                        v_sorted[row_inds, Mid_Right]) / 2
            inds_left = (w_flat < mid_vals)
            Right[inds_left] = Mid_Left[inds_left]
            Left[~inds_left] = Mid_Right[~inds_left]
        assignment_inds = inds[row_inds, Left].view(d1, d2)
        return bin_mat[assignment_inds, :]
    else:
        # w is d1xd2, Alpha is k
        v = bin_mat.mv(Alpha)
        v_sorted, inds = torch.sort(v)
        bin_mat_sorted = bin_mat[inds,:]
        # Binary search to find nearest neighbor
        Left = torch.zeros(w.shape, dtype=torch.long, device=w.device)
        Right = torch.ones(w.shape, dtype=torch.long, device=w.device) \
                * (2 ** n_bits - 1)
        for i in range(n_bits):
            Mid_Left = (Left + Right - 1) / 2
            Mid_Right = Mid_Left + 1
            mid_vals = (v_sorted[Mid_Left] + v_sorted[Mid_Right]) / 2
            inds_left = (w < mid_vals)
            Right[inds_left] = Mid_Left[inds_left]
            Left[~inds_left] = Mid_Right[~inds_left]
        return bin_mat_sorted[Left]


def find_B_numpy(w, Alpha, by_row=False):
    '''Find optimal quantization assignment via binary search'''
    n_bits = Alpha.shape[-1]
    bin_mat = BinaryMatrices[n_bits]
    if by_row:
        d1, d2 = w.shape
        row_inds = np.arange(d1).repeat(d2)
        # w is d1xd2, Alpha is d1xk, v is d1x2^k
        v = Alpha.dot(bin_mat.T)
        v_sorted, inds = np.sort(v), np.argsort(v)
        # Binary search to find nearest neighbor
        w_flat = w.flatten()
        Left = np.zeros(d1*d2, dtype=np.int32)
        Right = np.ones(d1*d2, dtype=np.int32) * (2 ** n_bits - 1)
        for i in range(n_bits):
            Mid_Left = (Left + Right - 1) // 2
            Mid_Right = Mid_Left + 1
            mid_vals = (v_sorted[row_inds, Mid_Left] + \
                        v_sorted[row_inds, Mid_Right]) / 2
            inds_left = (w_flat < mid_vals)
            Right[inds_left] = Mid_Left[inds_left]
            Left[~inds_left] = Mid_Right[~inds_left]
        assignment_inds = inds[row_inds, Left].reshape(d1, d2)
        return bin_mat[assignment_inds, :]
    else:
        # w is d1xd2, Alpha is k
        v = bin_mat.dot(Alpha)
        v_sorted, inds = np.sort(v), np.argsort(v)
        bin_mat_sorted = bin_mat[inds,:]
        # Binary search to find nearest neighbor
        Left = np.zeros(w.shape, dtype=np.int32)
        Right = np.ones(w.shape, dtype=np.int32) * (2 ** n_bits - 1)
        for i in range(n_bits):
            Mid_Left = (Left + Right - 1) // 2
            Mid_Right = Mid_Left + 1
            mid_vals = (v_sorted[Mid_Left] + v_sorted[Mid_Right]) / 2
            inds_left = (w < mid_vals)
            Right[inds_left] = Mid_Left[inds_left]
            Left[~inds_left] = Mid_Right[~inds_left]
        return bin_mat_sorted[Left]
