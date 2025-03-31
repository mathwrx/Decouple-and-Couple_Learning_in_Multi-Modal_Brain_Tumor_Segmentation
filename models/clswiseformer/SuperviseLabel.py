import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class SuperviseLabel(nn.Module):
    def __init__(self, item_future_num):
        super(SuperviseLabel, self).__init__()

        self.supervise_label_1 = nn.Conv3d(
            item_future_num,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.down_label_1 = nn.Conv3d(
            32,
            2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.supervise_label_2 = nn.Conv3d(
            item_future_num,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.down_label_2 = nn.Conv3d(
            32,
            2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.supervise_label_4 = nn.Conv3d(
            item_future_num,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.down_label_4 = nn.Conv3d(
            32,
            2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.sample_scale = 8
        self.Softmax_1 = nn.Softmax(dim=1)
        self.Softmax_2 = nn.Softmax(dim=1)
        self.Softmax_4 = nn.Softmax(dim=1)

    def forward(self, supervise_semantic_01, supervise_semantic_02, supervise_semantic_04):
        loss = {}
        supervise_semantic_01 = self.supervise_label_1(supervise_semantic_01)  # 1 32 8 8 8
        supervise_semantic_01 = self.down_label_1(supervise_semantic_01)  # 1 2 8 8 8
        supervise_semantic_01 = F.interpolate(supervise_semantic_01, scale_factor=self.sample_scale, mode='trilinear',
                                              align_corners=False)
        supervise_semantic_01 = self.Softmax_1(supervise_semantic_01)

        loss['01'] = supervise_semantic_01
        # label 2
        supervise_semantic_02 = self.supervise_label_2(supervise_semantic_02)  # 1 32 8 8 8
        supervise_semantic_02 = self.down_label_2(supervise_semantic_02)  # 1 2 8 8 8
        supervise_semantic_02 = F.interpolate(supervise_semantic_02, scale_factor=self.sample_scale, mode='trilinear',
                                              align_corners=False)
        supervise_semantic_02 = self.Softmax_2(supervise_semantic_02)
        loss['02'] = supervise_semantic_02
        # label 4
        supervise_semantic_04 = self.supervise_label_4(supervise_semantic_04)  # 1 128 8 8 8
        supervise_semantic_04 = self.down_label_4(supervise_semantic_04)  # 1 2 8 8 8
        supervise_semantic_04 = F.interpolate(supervise_semantic_04, scale_factor=self.sample_scale, mode='trilinear',
                                              align_corners=False)
        supervise_semantic_04 = self.Softmax_4(supervise_semantic_04)
        loss['04'] = supervise_semantic_04
        return loss
