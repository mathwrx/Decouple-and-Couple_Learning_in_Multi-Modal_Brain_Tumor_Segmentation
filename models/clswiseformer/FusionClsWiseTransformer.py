import torch
import torch.nn as nn

from models.clswiseformer.ResidualNorm import Residual, PreNormDrop, PreNorm, FeedForward
from models.clswiseformer.SelfAttention import DualSelfAttention


class FusionClsWiseTransformerModel(nn.Module):
    def __init__(
            self,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        self.compress_list = []
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.expand_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        mlp_dim,
                        dropout_rate,
                        DualSelfAttention(hidden_size=mlp_dim, num_heads=heads, dropout_rate=attn_dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(mlp_dim, FeedForward(mlp_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)
        #self.fusion_con = nn.Conv1d(128 * 3, 64 * 3, kernel_size=1)

    def forward(self, fusion_semantic):
        result_01 = self.cross_attention_list[0](fusion_semantic, fusion_semantic)

        # result_02 = self.cross_attention_list[0](fusion_semantic_04, fusion_semantic_1_2)
        #
        # result = torch.cat((result_01, result_02), dim=1)

        #result = self.fusion_con(result)
        #result = cross_1 + cross_2 + cross_3
        # FFN
        x = self.cross_ffn_list[0](result_01)
        return x
