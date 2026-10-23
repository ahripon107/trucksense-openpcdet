"""
Native-PyTorch fallback for the subset of torch_scatter used by the dynamic VFEs.
Only dim=0 with a 1-D index is supported.
"""
import torch


def _reduce(src, index, dim, reduce):
    assert dim == 0 and index.dim() == 1
    num = int(index.max()) + 1 if index.numel() > 0 else 0
    out = src.new_zeros((num,) + src.shape[1:])
    return out.index_reduce_(0, index, src, reduce, include_self=False)


def scatter_mean(src, index, dim=0):
    return _reduce(src, index, dim, 'mean')


def scatter_max(src, index, dim=0):
    # torch_scatter returns (values, argmax); callers here only use values
    return _reduce(src, index, dim, 'amax'), None
