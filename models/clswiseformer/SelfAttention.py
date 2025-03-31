import torch
import torch.nn as nn


class SingleSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
    ) -> None:
        super(SingleSelfAttention, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, _ = x.shape
        # b h (qkv l d) -> qkv b l h d
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        # b h l d -> b l (h d)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B, N, -1)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class DualSelfAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
    ) -> None:
        super(DualSelfAttention, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x, x2):
        # if x.shape != x2.shape:
        #     raise ValueError("x x2 shape not same")
        # x2
        B_2, N_2, _ = x2.shape
        # b h (qkv l d) -> qkv b l h d
        qkv_2 = self.qkv(x2)
        qkv_2 = qkv_2.reshape(B_2, N_2, 3, self.num_heads, -1)
        qkv_2 = qkv_2.permute(2, 0, 3, 1, 4).contiguous()
        #q = qkv_2[0]
        k = qkv_2[1]
        v = qkv_2[2]
        B, N, _ = x.shape
        # b h (qkv l d) -> qkv b l h d
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q = qkv[0]
        # k = qkv[1]
        # v = qkv[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        result = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        # b h l d -> b l (h d)
        result = result.permute(0, 2, 1, 3).contiguous()
        result = result.reshape(B, N, -1)
        result = self.out_proj(result)
        result = self.drop_output(result)
        return result

if __name__ == '__main__':
    in_channels = 128
    bach_size = 1
    patch_size = (2, 2, 2)
    image_size = (8, 8, 8)
    e_token_01 = torch.randn(bach_size, in_channels, image_size[0], image_size[1], image_size[2])


    e_token_01 = e_token_01.reshape(1, in_channels, 8 // patch_size[0], patch_size[0], 8 // patch_size[1],
                                    patch_size[1], 8 // patch_size[2], patch_size[2])
    e_token_01 = e_token_01.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    e_token_01 = e_token_01.flatten(1, 3)
    e_token_01 = e_token_01.flatten(2, 5)
    # ln = torch.nn.LayerNorm(1032)
    # ll = ln(e_token_01)
    # print(ll.shape)


    mid_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
    single = SingleSelfAttention(hidden_size=mid_dim, num_heads=1, dropout_rate=0.1)
    single(e_token_01)
