"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupMoLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
                 # TODO : original t = 0.07
        super(SupMoLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, labelqueue=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.zeros_like(features)
            mask = torch.scatter(
                torch.zeros_like(mask), 1, 
                torch.arange(batch_size).view(-1, 1).to(device), 1
            )
        else:
            mask = mask.float().to(device)

        # # compute logits
        # # for numerical stability
        logits_max, _ = torch.max(features, dim=1, keepdim=True)
        logits = features - logits_max.detach()

        # compute log_prob
        exp_pos_logits_sum = (logits * mask).sum(1)
        exp_all_logits_sum_log = torch.log(torch.exp(logits).sum(1))
        log_prob = exp_pos_logits_sum - exp_all_logits_sum_log

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = log_prob / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
