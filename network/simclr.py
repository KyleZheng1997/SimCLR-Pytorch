# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from network.backbone import *


class Simclr(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=128):
        super(Simclr, self).__init__()
        self.encoder_q = BackBone(dim=dim)


    def forward(self, x1, x2, rank):
        b = x1.size(0)

        feat = nn.functional.normalize(self.encoder_q(torch.cat([x1, x2])))
        other = concat_all_gather(feat, rank)

        prob = feat @ torch.cat([feat, other]).T / 0.1
        diagnal_mask = (1 - torch.eye(prob.size(0), prob.size(1))).bool().cuda()
        logits = torch.masked_select(prob, diagnal_mask).reshape(prob.size(0), -1)

        first_half_label = torch.arange(b-1, 2*b-1).long().cuda()
        second_half_label = torch.arange(0, b).long().cuda()
        labels = torch.cat([first_half_label, second_half_label])
        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor, rank):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    other = torch.cat(tensors_gather[:rank] + tensors_gather[rank+1:], dim=0)
    return other
