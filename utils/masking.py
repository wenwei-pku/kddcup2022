import torch
import math
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def log_mask(win_len, sub_len):
    mask = torch.zeros((win_len, win_len), dtype=torch.float)
    for i in range(win_len):
        mask[i] = row_mask(i, sub_len, win_len)
    return mask.view(1, 1, mask.size(0), mask.size(1))


def row_mask(index, sub_len, win_len):
    """
    Remark:
    1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
        should deal with CUDA kernel, which we haven't implemented yet.
    2 . Our default setting here use Local attention and Restart attention.
    3 . For index-th row, if its past is smaller than the number of cells the last
        cell can attend, we can allow current cell to attend all past cells to fully
        utilize parallel computing in dense matrices with sparse multiplication."""
    log_l = math.ceil(np.log2(sub_len))
    mask = torch.zeros((win_len), dtype=torch.float)
    if((win_len // sub_len) * 2 * (log_l) > index):
        mask[:(index + 1)] = 1
    else:
        while(index >= 0):
            if((index - log_l + 1) < 0):
                mask[:index] = 1
                break
            mask[index - log_l + 1:(index + 1)] = 1  # Local attention
            for i in range(0, log_l):
                new_index = index - log_l + 1 - 2**i
                if((index - new_index) <= sub_len and new_index >= 0):
                    mask[new_index] = 1
            index -= sub_len
    return mask
