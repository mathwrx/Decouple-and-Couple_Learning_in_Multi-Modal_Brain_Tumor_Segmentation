import torch
import torch.nn as nn
from models.clswiseformer.SelfAttention import DualSelfAttention
from models.clswiseformer.ResidualNorm import Residual, PreNormDrop, PreNorm, FeedForward


class TwoClsWiseTransformerModel(nn.Module):
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
                        DualSelfAttention(mlp_dim, num_heads=heads, dropout_rate=attn_dropout_rate),
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

    def forward(self, edge_fea, se_fea_supple, semantic_fea, supple_fea_edge):
        #edge_fea_01, se_fea_01_supple, semantic_fea_01, supple_fea_01_edge
        #fea_edge_self = self.cross_attention_list[0](edge_fea, edge_fea)
        edge_query_semantic = self.cross_attention_list[0](edge_fea, se_fea_supple)

        #fea_semantic_self = self.cross_attention_list[0](semantic_fea, semantic_fea)
        semantic_query_edge = self.cross_attention_list[0](semantic_fea, supple_fea_edge)

        result_edge = self.cross_attention_list[0](edge_query_semantic, semantic_query_edge)
        result_semantic = self.cross_attention_list[0](semantic_query_edge, edge_query_semantic)

        cross = torch.cat((result_edge, result_semantic), dim=1)
        # FFN
        x = self.cross_ffn_list[0](cross)
        return x
