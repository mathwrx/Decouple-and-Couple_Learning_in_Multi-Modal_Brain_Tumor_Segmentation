import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'ln':
        m = nn.LayerNorm(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='in'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class EnDownTo8(nn.Module):
    def __init__(self, in_channels, out_channels, stride=4):
        super(EnDownTo8, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y

class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnBlock1_1 = EnBlock(in_channels=base_channels)

        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        #self.EnDown1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=2, dilation=2)
        #self.EnDown_Tmp_1 = EnDownTo8(in_channels=base_channels * 2, out_channels=base_channels * 2, stride=2)
        #self.EnDown_Tmp_1 = nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1, dilation=17)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)
        #self.EnDown2 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=2, dilation=2)
        #self.EnDown_Tmp_2 = EnDownTo8(in_channels=base_channels * 4, out_channels=base_channels * 4, stride=1)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)

        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)

        #self.EnDown3 = nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, stride=1, padding=1, dilation=9)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnDown_4 = EnDownTo8(in_channels=base_channels * 8, out_channels=base_channels * 16, stride=1)
        # self.EnDown4 = EnDown(in_channels=base_channels*8, out_channels=base_channels*16)

        # self.EnBlock5_1 = EnBlock(in_channels=base_channels * 16)
        # self.EnBlock5_2 = EnBlock(in_channels=base_channels * 16)

    def forward(self, x):
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_1 = self.EnBlock1_1(x1_1)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)
        #x1_tmp = self.EnDown_Tmp_1(x2_1)


        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)
        #x2_tmp = self.EnDown_Tmp_2(x3_1)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_1 = self.EnBlock4_2(x4_1)
        x4_1 = self.EnDown_4(x4_1)
        # x4_2 = self.EnDown4(x4_1)  # (1, 256, 8, 8, 8)
        # x3_tmp = self.EnDown_Tmp_3(x4_1)

        # x5_1 = self.EnBlock5_1(x4_2)
        # x5_1 = self.EnBlock5_2(x5_1)  # (1, 256, 8, 8, 8)
        # print(x4_1.shape)
        # print(x2_tmp.shape)
        # print(x1_tmp.shape)
        # print(x3_1.shape)
        return x1_1, x2_1, x3_1,   x4_1


if __name__ == '__main__':
    with torch.no_grad():
        import os
        from thop import profile, clever_format
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128))
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet(in_channels=4, base_channels=16, num_classes=4)
        #model.cuda()
        #output = model(x)
        macs, params = profile(model, inputs=(x, ))
        macs, params = clever_format([macs, params], "%.3f")
        print('FLOPS:', macs)
        print('Params:', params)
