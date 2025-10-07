#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

# Total Variation 2D for regularization
def tv2d(prob_map: torch.Tensor) -> torch.Tensor:
    # Differences along height
    dx = prob_map[:, :, 1:, :] - prob_map[:, :, :-1, :]
    # Differences along width
    dy = prob_map[:, :, :, 1:] - prob_map[:, :, :, :-1]

    # L1 penalty
    return dx.abs().mean() + dy.abs().mean()

def tv2d_foreground(pred_softmax: torch.Tensor, fg_classes=[1,2,3,4]) -> torch.Tensor:
    # Sum probabilities of all foreground classes
    p_fg = pred_softmax[:, fg_classes, ...].sum(dim=1, keepdim=True)  # [B,1,H,W]

    # Apply TV on the foreground probability map
    return tv2d(p_fg)

class CEWithTV:
    def __init__(self, lam_tv=1e-3, num_classes=5):
        # Full CE over all classes
        self.ce = CrossEntropy(idk=list(range(num_classes)))
        self.lam_tv = lam_tv
        self.fg_classes = list(range(1, num_classes))  # exclude background

    def __call__(self, pred_softmax, weak_target):
        # 1) Standard full CE loss
        loss = self.ce(pred_softmax, weak_target)

        # 2) Add foreground TV regularization
        if self.lam_tv > 0:
            tv_term = tv2d_foreground(pred_softmax, fg_classes=self.fg_classes)
            loss = loss + self.lam_tv * tv_term

        return loss
