import torch.nn as nn
import torch.nn.functional as F


class EdgeSuperviseLabel(nn.Module):
    def __init__(self, item_future_num):
        super(EdgeSuperviseLabel, self).__init__()

        self.edge_supervise_label_1 = nn.Conv3d(
            item_future_num,
            8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.edge_down_label_1 = nn.Conv3d(
            8,
            2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.edge_supervise_label_2 = nn.Conv3d(
            item_future_num,
            8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.edge_down_label_2 = nn.Conv3d(
            8,
            2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.edge_supervise_label_4 = nn.Conv3d(
            item_future_num,
            8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.edge_down_label_4 = nn.Conv3d(
            8,
            2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.sample_scale = 4
        self.edge_softmax_1 = nn.Softmax(dim=1)
        self.edge_softmax_2 = nn.Softmax(dim=1)
        self.edge_softmax_4 = nn.Softmax(dim=1)

    def forward(self, supervise_edge_01, supervise_edge_02, supervise_edge_04):
        loss = {}
        supervise_edge_01 = self.edge_supervise_label_1(supervise_edge_01)  # 1 32 8 8 8
        supervise_edge_01 = self.edge_down_label_1(supervise_edge_01)  # 1 2 8 8 8
        supervise_edge_01 = F.interpolate(supervise_edge_01, scale_factor=self.sample_scale, mode='trilinear', align_corners=False)
        supervise_edge_01 = self.edge_softmax_1(supervise_edge_01)

        loss['01'] = supervise_edge_01
        # label 2
        supervise_edge_02 = self.edge_supervise_label_2(supervise_edge_02)  # 1 32 8 8 8
        supervise_edge_02 = self.edge_down_label_2(supervise_edge_02)  # 1 2 8 8 8
        supervise_edge_02 = F.interpolate(supervise_edge_02, scale_factor=self.sample_scale, mode='trilinear', align_corners=False)
        supervise_edge_02 = self.edge_softmax_2(supervise_edge_02)
        loss['02'] = supervise_edge_02
        # label 4
        supervise_edge_04 = self.edge_supervise_label_4(supervise_edge_04)  # 1 128 8 8 8
        supervise_edge_04 = self.edge_down_label_4(supervise_edge_04)  # 1 2 8 8 8
        supervise_edge_04 = F.interpolate(supervise_edge_04, scale_factor=self.sample_scale, mode='trilinear', align_corners=False)
        supervise_edge_04 = self.edge_softmax_4(supervise_edge_04)
        loss['04'] = supervise_edge_04
        return loss
