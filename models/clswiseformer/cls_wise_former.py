import torch
import torch.nn as nn
from models.clswiseformer.ClsWiseTransformer import TwoClsWiseTransformerModel
from models.clswiseformer.PositionalEncoding import FixedPositionalEncoding, ExtendFixedPositionalEncoding, \
    LearnedPositionalEncoding
from models.clswiseformer.Unet_skipconnection import Unet
import torch.nn.functional as F
from models.clswiseformer.SuperviseLabel import SuperviseLabel
from models.clswiseformer.EdgeSuperviseLabel import EdgeSuperviseLabel
from models.clswiseformer.FusionClsWiseTransformer import FusionClsWiseTransformerModel
import numpy as np
import os


def convert_dim(fea, image_size, patch_size):
    fea = fea.reshape(fea.size(0), fea.size(1),
                      image_size[0] // patch_size[0], patch_size[0],
                      image_size[1] // patch_size[1], patch_size[1],
                      image_size[2] // patch_size[2], patch_size[2])
    fea = fea.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    fea = fea.flatten(1, 3)
    fea = fea.flatten(2, 5)
    return fea


def split_dim(fea, in_channel, image_size, patch_size):
    fea = fea.reshape(fea.size(0),  # 0
                      image_size[0] // patch_size[0],  # 1
                      image_size[1] // patch_size[1],  # 2
                      image_size[2] // patch_size[2],  # 3
                      in_channel,  # 4
                      patch_size[0],  # 5
                      patch_size[1],  # 6
                      patch_size[2])  # 7
    fea = fea.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    fea = fea.flatten(2, 3)
    fea = fea.flatten(3, 4)
    fea = fea.flatten(4, 5)
    return fea


class ClsWiseFormer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
            gpu=0
    ):
        super(ClsWiseFormer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation
        self.feature_num = 3
        self.item_feature_n = 128
        self.feature_sum = self.item_feature_n * self.feature_num
        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.patch_size = (2, 2, 1)
        self.image_size = (16, 16, 16)
        self.edge_feature_n = 32
        self.top_num = 128
        self.select_num_1 = 85
        self.select_num_2 = 85
        self.select_num_4 = 86
        self.edge_image_size = (32, 32, 32)
        self.edge_patch_size = (4, 2, 2)
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(self.image_size, self.patch_size)])
        if positional_encoding_type == "learned":
            self.label_01_position_encoding = LearnedPositionalEncoding(129, 512)
            self.label_02_position_encoding = LearnedPositionalEncoding(129, 512)
            self.label_04_position_encoding = LearnedPositionalEncoding(129, 512)
        elif positional_encoding_type == "fixed":
            self.label_01_position_encoding = ExtendFixedPositionalEncoding(
                self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                self.n_patches
            )
            self.label_02_position_encoding = ExtendFixedPositionalEncoding(
                self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                self.n_patches
            )
            self.label_04_position_encoding = ExtendFixedPositionalEncoding(
                self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                self.n_patches
            )
        self.transformer_01_drop = nn.Dropout(p=self.dropout_rate)
        self.transformer_01 = TwoClsWiseTransformerModel(
            depth=1,
            heads=self.num_heads,
            mlp_dim=self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        self.transformer_02_drop = nn.Dropout(p=self.dropout_rate)
        self.transformer_02 = TwoClsWiseTransformerModel(
            depth=1,
            heads=self.num_heads,
            mlp_dim=self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        self.transformer_04_drop = nn.Dropout(p=self.dropout_rate)
        self.transformer_04 = TwoClsWiseTransformerModel(
            depth=1,
            heads=self.num_heads,
            mlp_dim=self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )

        self.transformer_fusion_drop = nn.Dropout(p=self.dropout_rate)
        self.fusion_label_pos = ExtendFixedPositionalEncoding(
            self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            self.n_patches)
        self.fusion_transformer_1_2_4 = FusionClsWiseTransformerModel(
            depth=1,
            heads=self.num_heads,
            mlp_dim=self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )

        # self.conv_edge_fea_fu = nn.Conv3d(
        #     256,
        #     self.feature_sum,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )

        # self.conv_semantic_fu = nn.Conv3d(
        #     256,
        #     self.feature_sum,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )

        self.conv_semantic_1 = nn.Conv3d(
            256,
            self.item_feature_n,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv_semantic_2 = nn.Conv3d(
            256,
            self.item_feature_n,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv_semantic_4 = nn.Conv3d(
            256,
            self.item_feature_n,
            kernel_size=3,
            stride=1,
            padding=1
        )


        self.conv_mid_fea_1 = nn.Conv3d(
            96,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv_mid_fea_2 = nn.Conv3d(
            96,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv_mid_fea_4 = nn.Conv3d(
            96,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.Unet_list = Unet(in_channels=4, base_channels=16, num_classes=4)
        self.bn_list = nn.InstanceNorm3d(self.item_feature_n)
        self.relu_list = nn.LeakyReLU(inplace=True)

        self.bn_list_2 = nn.InstanceNorm3d(self.item_feature_n)
        self.relu_list_2 = nn.LeakyReLU(inplace=True)

        self.bn_list_4 = nn.InstanceNorm3d(self.item_feature_n)
        self.relu_list_4 = nn.LeakyReLU(inplace=True)

        self.bn_edge = nn.InstanceNorm3d(self.edge_feature_n)
        self.relu_edge = nn.LeakyReLU(inplace=True)

        self.bn_edge_2 = nn.InstanceNorm3d(self.edge_feature_n)
        self.relu_edge_2 = nn.LeakyReLU(inplace=True)

        self.bn_edge_4 = nn.InstanceNorm3d(self.edge_feature_n)
        self.relu_edge_4 = nn.LeakyReLU(inplace=True)

        self.decoder = Decoder(self.embedding_dim, num_classes)
        self.supervise_label = SuperviseLabel(self.item_feature_n)
        self.edge_supervise_label = EdgeSuperviseLabel(self.edge_feature_n)

        self.mid_supervise_label = SuperviseLabel(self.item_feature_n)
        self.mid_edge_supervise_label = EdgeSuperviseLabel(self.edge_feature_n)

        self.e_token_01 = nn.Parameter(
            torch.zeros(1, 1, self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))
        self.s_token_01 = nn.Parameter(
            torch.zeros(1, 1, self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))

        self.e_token_02 = nn.Parameter(
            torch.zeros(1, 1, self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))
        self.s_token_02 = nn.Parameter(
            torch.zeros(1, 1, self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))

        self.e_token_04 = nn.Parameter(
            torch.zeros(1, 1, self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))
        self.s_token_04 = nn.Parameter(
            torch.zeros(1, 1, self.item_feature_n * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))

        # self.result_fusion = nn.Conv1d(64 * 3, 64, kernel_size=1)
        torch.nn.init.trunc_normal_(self.e_token_01, std=0.02)
        torch.nn.init.trunc_normal_(self.s_token_01, std=0.02)

        torch.nn.init.trunc_normal_(self.e_token_02, std=0.02)
        torch.nn.init.trunc_normal_(self.s_token_02, std=0.02)

        torch.nn.init.trunc_normal_(self.e_token_04, std=0.02)
        torch.nn.init.trunc_normal_(self.s_token_04, std=0.02)

        self.sum_fusion = nn.Conv3d(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # self.fusion_1_2_4 = nn.Conv3d(
        #     self.feature_sum,
        #     self.item_feature_n,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )

        self.conv_64_to_32 = torch.nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)

        path = '2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training'
        with open(os.path.join(path, 'fix_index.txt')) as f:
            lines = f.readlines()
            self.dict_index = eval(lines[0])

    def encode(self, x, missing_modal):

        x1_1, x2_1, x3_1, x = self.Unet_list(x)
        # edge
        x2_1_tmp = self.conv_64_to_32(x2_1)
        x_2_3 = torch.cat((x2_1_tmp, x3_1), dim=1)
        mid_fea_1 = self.conv_mid_fea_1(x_2_3)
        mid_fea_1 = self.bn_edge(mid_fea_1)
        mid_fea_1 = self.relu_edge(mid_fea_1)

        mid_fea_2 = self.conv_mid_fea_2(x_2_3)
        mid_fea_2 = self.bn_edge_2(mid_fea_2)
        mid_fea_2 = self.relu_edge_2(mid_fea_2)

        mid_fea_4 = self.conv_mid_fea_4(x_2_3)
        mid_fea_4 = self.bn_edge_4(mid_fea_4)
        mid_fea_4 = self.relu_edge_4(mid_fea_4)

        edge_1_2_4 = [mid_fea_1,
                      mid_fea_2,
                      mid_fea_4]

        # edge_1_2_4 = [mid_fea[:, 0:self.edge_feature_n, ...],
        #               mid_fea[:, self.edge_feature_n: 2 * self.edge_feature_n, ...],
        #               mid_fea[:, 2 * self.edge_feature_n:self.edge_feature_n * self.feature_num, ...]]

        # semantic
        # x = self.conv_semantic_fu(x)
        # x = self.bn_list(x)
        # x = self.relu_list(x)
        # se_1_2_4 = [x[:, 0:self.item_feature_n, ...],
        #             x[:, self.item_feature_n: 2 * self.item_feature_n, ...],
        #             x[:, 2 * self.item_feature_n:self.feature_sum, ...]]

        fea_1 = self.conv_semantic_1(x)
        fea_1 = self.bn_list(fea_1)
        fea_1 = self.relu_list(fea_1)

        fea_2 = self.conv_semantic_2(x)
        fea_2 = self.bn_list_2(fea_2)
        fea_2 = self.relu_list_2(fea_2)

        fea_4 = self.conv_semantic_4(x)
        fea_4 = self.bn_list_4(fea_4)
        fea_4 = self.relu_list_4(fea_4)

        se_1_2_4 = [fea_1,
                    fea_2,
                    fea_4]


        # mid supervise
        mid_semantic_loss = self.mid_supervise_label(se_1_2_4[0], se_1_2_4[1], se_1_2_4[2])
        mid_edge_loss = self.mid_edge_supervise_label(edge_1_2_4[0], edge_1_2_4[1], edge_1_2_4[2])

        # edge semantic
        edge_semantic_01 = [edge_1_2_4[0], se_1_2_4[0]]
        edge_semantic_02 = [edge_1_2_4[1], se_1_2_4[1]]
        edge_semantic_04 = [edge_1_2_4[2], se_1_2_4[2]]

        # label 1
        edge_semantic_01[0] = convert_dim(edge_semantic_01[0], self.edge_image_size, self.edge_patch_size)
        edge_semantic_01[1] = convert_dim(edge_semantic_01[1], self.image_size, self.patch_size)

        # select top_k
        top_k_1_edge = self.e_token_01 @ edge_semantic_01[0].transpose(2, 1).contiguous()
        _, top_k_1_index_edge = top_k_1_edge.topk(self.top_num, dim=2, largest=True, sorted=True)
        tmp_cat_1_edge = torch.index_select(edge_semantic_01[0], dim=1, index=top_k_1_index_edge[0, 0, ...])
        tmp_cat_1_edge = self.label_01_position_encoding(tmp_cat_1_edge)
        tmp_cat_1_edge = self.transformer_01_drop(tmp_cat_1_edge)
        edge_fea_01 = torch.cat((self.e_token_01, tmp_cat_1_edge), dim=1)
        # select supplement information
        top_k_1_e_supple = self.e_token_01 @ edge_semantic_01[1].transpose(2, 1).contiguous()
        _, top_k_1_supple_index = top_k_1_e_supple.topk(self.top_num, dim=2, largest=True, sorted=True)
        supple_cat_1_se = torch.index_select(edge_semantic_01[1], dim=1, index=top_k_1_supple_index[0, 0, ...])
        supple_cat_1_se = self.label_01_position_encoding(supple_cat_1_se)
        supple_cat_1_se = self.transformer_01_drop(supple_cat_1_se)
        se_fea_01_supple = torch.cat((self.s_token_01, supple_cat_1_se), dim=1)

        # select top k semantic
        top_k_1_semantic = self.s_token_01 @ edge_semantic_01[1].transpose(2, 1).contiguous()
        _, top_k_1_index_semantic = top_k_1_semantic.topk(self.top_num, dim=2, largest=True, sorted=True)

        tmp_cat_1_semantic = torch.index_select(edge_semantic_01[1], dim=1, index=top_k_1_index_semantic[0, 0, ...])

        tmp_cat_1_semantic = self.label_01_position_encoding(tmp_cat_1_semantic)
        tmp_cat_1_semantic = self.transformer_01_drop(tmp_cat_1_semantic)
        semantic_fea_01 = torch.cat((self.s_token_01, tmp_cat_1_semantic), dim=1)

        # select edge supplement information
        top_k_1_s_supple = self.s_token_01 @ edge_semantic_01[0].transpose(2, 1).contiguous()
        _, top_k_1_s_index_supple = top_k_1_s_supple.topk(self.top_num, dim=2, largest=True, sorted=True)
        supple_cat_1_edge = torch.index_select(edge_semantic_01[0], dim=1, index=top_k_1_s_index_supple[0, 0, ...])

        supple_cat_1_edge = self.label_01_position_encoding(supple_cat_1_edge)
        supple_cat_1_edge = self.transformer_01_drop(supple_cat_1_edge)
        supple_fea_01_edge = torch.cat((self.e_token_01, supple_cat_1_edge), dim=1)

        # trans 1
        result_01 = self.transformer_01(edge_fea_01, se_fea_01_supple, semantic_fea_01, supple_fea_01_edge)

        # label 2
        edge_semantic_02[0] = convert_dim(edge_semantic_02[0], self.edge_image_size, self.edge_patch_size)
        edge_semantic_02[1] = convert_dim(edge_semantic_02[1], self.image_size, self.patch_size)

        # select top_k
        top_k_2_edge = self.e_token_02 @ edge_semantic_02[0].transpose(2, 1).contiguous()
        _, top_k_2_index_edge = top_k_2_edge.topk(self.top_num, dim=2, largest=True, sorted=True)
        tmp_cat_2_edge = torch.index_select(edge_semantic_02[0], dim=1, index=top_k_2_index_edge[0, 0, ...])

        tmp_cat_2_edge = self.label_02_position_encoding(tmp_cat_2_edge)
        tmp_cat_2_edge = self.transformer_02_drop(tmp_cat_2_edge)
        edge_fea_02 = torch.cat((self.e_token_02, tmp_cat_2_edge), dim=1)
        # supplement top k semantic feature
        topk_2_supple_se = self.e_token_02 @ edge_semantic_02[1].transpose(2, 1).contiguous()
        _, topk_2_index_supple_se = topk_2_supple_se.topk(self.top_num, dim=2, largest=True, sorted=True)
        supple_cat_2_se = torch.index_select(edge_semantic_02[1], dim=1, index=topk_2_index_supple_se[0, 0, ...])
        supple_cat_2_se = self.label_02_position_encoding(supple_cat_2_se)
        supple_cat_2_se = self.transformer_02_drop(supple_cat_2_se)
        supple_se_fea_02 = torch.cat((self.s_token_02, supple_cat_2_se), dim=1)

        # select top k semantic
        top_k_2_semantic = self.s_token_02 @ edge_semantic_02[1].transpose(2, 1).contiguous()
        _, top_k_2_index_semantic = top_k_2_semantic.topk(self.top_num, dim=2, largest=True, sorted=True)
        tmp_cat_2_semantic = torch.index_select(edge_semantic_02[1], dim=1, index=top_k_2_index_semantic[0, 0, ...])
        tmp_cat_2_semantic = self.label_02_position_encoding(tmp_cat_2_semantic)
        tmp_cat_2_semantic = self.transformer_02_drop(tmp_cat_2_semantic)
        semantic_fea_02 = torch.cat((self.s_token_02, tmp_cat_2_semantic), dim=1)

        # supplement top k edge feature
        topk_2_supple_edge = self.s_token_02 @ edge_semantic_02[0].transpose(2, 1).contiguous()
        _, topk_2_index_edge_supple = topk_2_supple_edge.topk(self.top_num, dim=2, largest=True, sorted=True)
        supple_cat_2_edge = torch.index_select(edge_semantic_02[0], dim=1, index=topk_2_index_edge_supple[0, 0, ...])
        supple_cat_2_edge = self.label_02_position_encoding(supple_cat_2_edge)
        supple_cat_2_edge = self.transformer_02_drop(supple_cat_2_edge)
        supple_edge_fea_02 = torch.cat((self.e_token_02, supple_cat_2_edge), dim=1)
        # trans 2
        result_02 = self.transformer_02(edge_fea_02, supple_se_fea_02, semantic_fea_02, supple_edge_fea_02)

        # label 4
        edge_semantic_04[0] = convert_dim(edge_semantic_04[0], self.edge_image_size, self.edge_patch_size)
        edge_semantic_04[1] = convert_dim(edge_semantic_04[1], self.image_size, self.patch_size)
        # select top_k
        top_k_4_edge = self.e_token_04 @ edge_semantic_04[0].transpose(2, 1).contiguous()
        _, top_k_4_index_edge = top_k_4_edge.topk(self.top_num, dim=2, largest=True, sorted=True)
        tmp_cat_4_edge = torch.index_select(edge_semantic_04[0], dim=1, index=top_k_4_index_edge[0, 0, ...])
        tmp_cat_4_edge = self.label_04_position_encoding(tmp_cat_4_edge)
        tmp_cat_4_edge = self.transformer_04_drop(tmp_cat_4_edge)
        edge_fea_04 = torch.cat((self.e_token_04, tmp_cat_4_edge), dim=1)

        # supplement top k semantic feature
        topk_4_spple_se = self.e_token_04 @ edge_semantic_04[1].transpose(2, 1).contiguous()
        _, topk_4_index_se_supple = topk_4_spple_se.topk(self.top_num, dim=2, largest=True, sorted=True)
        supple_cat_4_se = torch.index_select(edge_semantic_04[1], dim=1, index=topk_4_index_se_supple[0, 0, ...])
        supple_cat_4_se = self.label_04_position_encoding(supple_cat_4_se)
        supple_cat_4_se = self.transformer_04_drop(supple_cat_4_se)
        supple_se_fea_04 = torch.cat((self.s_token_04, supple_cat_4_se), dim=1)

        # select top k semantic
        top_k_4_semantic = self.s_token_04 @ edge_semantic_04[1].transpose(2, 1).contiguous()
        _, top_k_4_index_semantic = top_k_4_semantic.topk(self.top_num, dim=2, largest=True, sorted=True)
        tmp_cat_4_semantic = torch.index_select(edge_semantic_04[1], dim=1, index=top_k_4_index_semantic[0, 0, ...])
        tmp_cat_4_semantic = self.label_04_position_encoding(tmp_cat_4_semantic)
        tmp_cat_4_semantic = self.transformer_04_drop(tmp_cat_4_semantic)
        semantic_fea_04 = torch.cat((self.s_token_04, tmp_cat_4_semantic), dim=1)

        # supplement top k edge feature
        topk_4_supple_edge = self.s_token_04 @ edge_semantic_04[0].transpose(2, 1).contiguous()
        _, topk_4_index_supple_edge = topk_4_supple_edge.topk(self.top_num, dim=2, largest=True, sorted=True)
        supple_cat_4_edge = torch.index_select(edge_semantic_04[0], dim=1, index=topk_4_index_supple_edge[0, 0, ...])
        supple_cat_4_edge = self.label_04_position_encoding(supple_cat_4_edge)
        supple_cat_4_edge = self.transformer_04_drop(supple_cat_4_edge)
        supple_edge_fea_04 = torch.cat((self.e_token_04, supple_cat_4_edge), dim=1)

        # trans 4
        result_04 = self.transformer_04(edge_fea_04, supple_se_fea_04, semantic_fea_04, supple_edge_fea_04)

        # label 1
        index_num = self.top_num + 1
        result_01_edge_fea = result_01[:, 0:index_num, ...]
        result_01_edge_token = result_01_edge_fea[:, 0:1, ...]
        result_01_edge_s = result_01_edge_fea[:, 1:index_num, ...]
        result_edge_index_01 = []
        for cat_index_1_edge in range(self.top_num):
            tmp_top_index_1 = top_k_1_index_edge[0, 0, cat_index_1_edge]
            result_edge_index_01.append(self.dict_index[str(tmp_top_index_1.item())])

        edge_semantic_01[0][0, ...].scatter_(0, torch.tensor(result_edge_index_01, device=edge_semantic_01[0].device),
                                             result_01_edge_s[0, ...])

        result_01_semantic_fea = result_01[:, index_num:index_num * 2, ...]
        result_01_semantic_token = result_01_semantic_fea[:, 0:1, ...]
        result_01_semantic_s = result_01_semantic_fea[:, 1:index_num, ...]
        result_semantic_index_01 = []
        for cat_index_1_semantic in range(self.top_num):
            tmp_top_index_1_semantic = top_k_1_index_semantic[0, 0, cat_index_1_semantic]
            result_semantic_index_01.append(self.dict_index[str(tmp_top_index_1_semantic.item())])
        edge_semantic_01[1][0, ...].scatter_(0,
                                             torch.tensor(result_semantic_index_01, device=edge_semantic_01[1].device),
                                             result_01_semantic_s[0, ...])

        supervise_edge_01 = result_01_edge_token * edge_semantic_01[0]
        supervise_edge_01 = split_dim(supervise_edge_01, self.edge_feature_n, self.edge_image_size, self.edge_patch_size)

        supervise_semantic_01 = result_01_semantic_token * edge_semantic_01[1]
        supervise_semantic_01 = split_dim(supervise_semantic_01, self.item_feature_n, self.image_size, self.patch_size)

        # label 2
        result_02_edge_fea = result_02[:, 0:index_num, ...]
        result_02_edge_token = result_02_edge_fea[:, 0:1, ...]
        result_02_edge_s = result_02_edge_fea[:, 1:index_num, ...]
        result_edge_index_02 = []
        for cat_index_2_edge in range(self.top_num):
            tmp_top_index_2 = top_k_2_index_edge[0, 0, cat_index_2_edge]
            result_edge_index_02.append(self.dict_index[str(tmp_top_index_2.item())])
        edge_semantic_02[0][0, ...].scatter_(0, torch.tensor(result_edge_index_02, device=edge_semantic_02[0].device),
                                             result_02_edge_s[0, ...])

        result_02_semantic_fea = result_02[:, index_num:(index_num * 2), ...]
        result_02_semantic_token = result_02_semantic_fea[:, 0:1, ...]
        result_02_semantic_s = result_02_semantic_fea[:, 1:index_num, ...]
        result_semantic_index_02 = []
        for cat_index_2_semantic in range(self.top_num):
            tmp_top_index_2_semantic = top_k_2_index_semantic[0, 0, cat_index_2_semantic]
            result_semantic_index_02.append(self.dict_index[str(tmp_top_index_2_semantic.item())])
        edge_semantic_02[1][0, ...].scatter_(0, torch.tensor(result_semantic_index_02,
                                                             device=edge_semantic_02[1].device),
                                             result_02_semantic_s[0, ...])

        supervise_edge_02 = result_02_edge_token * edge_semantic_02[0]
        supervise_edge_02 = split_dim(supervise_edge_02, self.edge_feature_n, self.edge_image_size, self.edge_patch_size)

        supervise_semantic_02 = result_02_semantic_token * edge_semantic_02[1]
        supervise_semantic_02 = split_dim(supervise_semantic_02, self.item_feature_n, self.image_size, self.patch_size)

        # label 4
        result_04_edge_fea = result_04[:, 0:index_num, ...]
        result_04_edge_token = result_04_edge_fea[:, 0:1, ...]
        result_04_edge_s = result_04_edge_fea[:, 1:index_num, ...]
        result_edge_index_04 = []
        for cat_index_4_edge in range(self.top_num):
            tmp_top_index_4 = top_k_4_index_edge[0, 0, cat_index_4_edge]
            result_edge_index_04.append(self.dict_index[str(tmp_top_index_4.item())])
        edge_semantic_04[0][0, ...].scatter_(0, torch.tensor(result_edge_index_04,
                                                             device=edge_semantic_04[0].device),
                                             result_04_edge_s[0, ...])

        result_04_semantic_fea = result_04[:, index_num:(index_num * 2), ...]
        result_04_semantic_token = result_04_semantic_fea[:, 0:1, ...]
        result_04_semantic_s = result_04_semantic_fea[:, 1:index_num, ...]
        result_semantic_index_04 = []
        for cat_index_4_semantic in range(self.top_num):
            tmp_top_index_4_semantic = top_k_4_index_semantic[0, 0, cat_index_4_semantic]
            result_semantic_index_04.append(self.dict_index[str(tmp_top_index_4_semantic.item())])

        edge_semantic_04[1][0, ...].scatter_(0, torch.tensor(result_semantic_index_04,
                                                             device=edge_semantic_04[1].device),
                                             result_04_semantic_s[0, ...])

        supervise_edge_04 = result_04_edge_token * edge_semantic_04[0]
        supervise_edge_04 = split_dim(supervise_edge_04, self.edge_feature_n, self.edge_image_size, self.edge_patch_size)

        supervise_semantic_04 = result_04_semantic_token * edge_semantic_04[1]
        supervise_semantic_04 = split_dim(supervise_semantic_04, self.item_feature_n, self.image_size, self.patch_size)

        supervise_loss = self.supervise_label(supervise_semantic_01, supervise_semantic_02, supervise_semantic_04)
        edge_loss = self.edge_supervise_label(supervise_edge_01, supervise_edge_02, supervise_edge_04)

        # fusion label 1, fusion label 2, fusion label 4
        fusion_token = result_01_semantic_token + result_02_semantic_token + result_04_semantic_token
        fusion_feature = edge_semantic_01[1] + edge_semantic_02[1] + edge_semantic_04[1]

        top_k_fusion = fusion_token @ fusion_feature.transpose(2, 1).contiguous()
        _, fusion_index = top_k_fusion.topk(self.top_num, dim=2, largest=True, sorted=True)

        fusion_cat_semantic = torch.index_select(fusion_feature, dim=1, index=fusion_index[0, 0, ...])

        fusion_cat_semantic = self.fusion_label_pos(fusion_cat_semantic)
        fusion_cat_semantic = self.transformer_fusion_drop(fusion_cat_semantic)

        fusion_cat_semantic = torch.cat((fusion_token, fusion_cat_semantic), dim=1)

        result_1_2_4 = self.fusion_transformer_1_2_4(fusion_cat_semantic)

        # fusion label 1, fusion label 2, fusion label 4
        result_fusion = fusion_feature.clone()
        cross_fusion = result_1_2_4[:, 0:index_num, ...]
        cross_fusion_token = cross_fusion[:, 0:1, ...]
        cross_fusion_semantic = cross_fusion[:, 1:index_num, ...]
        result_fusion_index = []
        for result_cat_fusion_index in range(self.top_num):
            result_fusion_index_semantic = fusion_index[0, 0, result_cat_fusion_index]
            result_fusion_index.append(self.dict_index[str(result_fusion_index_semantic.item())])

        result_fusion[0, ...].scatter_(0, torch.tensor(result_fusion_index,
                                                       device=result_fusion.device),
                                       cross_fusion_semantic[0, ...])
        result_fusion = cross_fusion_token * result_fusion

        x = split_dim(result_fusion, self.item_feature_n, self.image_size, self.patch_size)

        # x = torch.cat((result_cross_1, result_cross_2, result_cross_4), dim=1)
        x = self.sum_fusion(x)
        return x1_1, x2_1, x3_1, x, supervise_loss, edge_loss, mid_semantic_loss, mid_edge_loss

    def forward(self, x, missing_modal):

        x1_1, x2_1, x3_1, encoder_output, supervise_loss, edge_loss, mid_semantic_loss, mid_edge_loss = self.encode(
            x, missing_modal)

        decoder_output = self.decoder(x1_1, x2_1, x3_1, None, encoder_output)

        return decoder_output, supervise_loss, edge_loss, mid_semantic_loss, mid_edge_loss

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)

        self.down_channel = nn.Conv3d(self.embedding_dim, self.embedding_dim // 2, kernel_size=1)
        self.Enblock8_1 = EnBlock2(in_channels=self.embedding_dim // 2)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 2)

        # self.DeUp5 = DeUp_Cat(in_channels=self.embedding_dim, out_channels=self.embedding_dim // 2)
        # self.DeBlock5 = DeBlock(in_channels=self.embedding_dim // 2)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 2, out_channels=self.embedding_dim // 4)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 4)
        self.DeBlock4_1 = DeBlock(in_channels=self.embedding_dim // 4)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 8)
        self.DeBlock3_1 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 16)
        self.DeBlock2_1 = DeBlock(in_channels=self.embedding_dim // 16)

        self.endconv = nn.Conv3d(self.embedding_dim // 16, self.num_classes, kernel_size=1)

    def forward(self, x1_1, x2_1, x3_1, x4_1, x):
        x8 = x
        x8 = self.down_channel(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)

        y4 = self.DeUp4(x8, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)
        y4 = self.DeBlock4_1(y4)

        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)
        y3 = self.DeBlock3_1(y3)

        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y2 = self.DeBlock2_1(y2)

        y = self.endconv(y2)  # (1, 4, 128, 128, 128)
        y = self.Softmax(y)
        return y


class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        # self.bn1 = nn.BatchNorm3d(in_channels // 2)
        self.bn1 = nn.InstanceNorm3d(in_channels // 2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        # self.bn2 = nn.BatchNorm3d(in_channels // 2)
        self.bn2 = nn.InstanceNorm3d(in_channels // 2)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        # self.bn2 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        # self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


def get_cls_wise_former(dataset='brats', _conv_repr=True, _pe_type="learned", gpu=0):
    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    num_channels = 4
    patch_dim = 16
    model = ClsWiseFormer(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=256,
        num_heads=8,
        num_layers=1,
        hidden_dim=2048,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
        gpu=gpu
    )

    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        from thop import profile, clever_format
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        #x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        x = torch.rand((1, 4, 128, 128, 128))
        # conv = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        # jff = conv(x)
        model = get_cls_wise_former(dataset='brats', _conv_repr=True, _pe_type="fixed", gpu=0)
        #model.cuda()
        #output = model(x, None)
        # print('output:', output.shape)
        print('xxxxxx')
        macs, params = profile(model, inputs=(x,None))
        macs, params = clever_format([macs, params], "%.3f")
        print('FLOPS:', macs)
        print('Params:', params)
