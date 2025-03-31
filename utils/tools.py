import torch.distributed as dist
import torch
import numpy as np
import torch.nn.functional as F
from medpy import metric


def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:, i, :, :, :] * target[:, i, :, :, :])
        l = torch.sum(output[:, i, :, :, :])
        r = torch.sum(target[:, i, :, :, :])
        if i == 0:
            dice = 2.0 * num / (l + r + eps)
        else:
            dice += 2.0 * num / (l + r + eps)
    return 1.0 - 1.0 * dice / num_cls


def softmax_weighted_loss(output, target, num_cls=4):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1, 2, 3)) * 1.0 / torch.sum(target, (1, 2, 3, 4)))
        weighted = torch.reshape(weighted, (-1, 1, 1, 1)).repeat(1, H, W, Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor


def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num / den


def mIOU(o, t, eps=1e-8):
    num = (o * t).sum() + eps
    den = (o | t).sum() + eps
    return num / den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output == 1), t=(target == 1)))
    mIOU_score.append(mIOU(o=(output == 2), t=(target == 2)))
    mIOU_score.append(mIOU(o=(output == 3), t=(target == 3)))
    return mIOU_score


def softmax_hd_dice(output, target):
    ret = []
    hd_ret = []
    # whole
    o = output > 0
    t = target > 0  # ce
    wt = dice_score(o, t)
    ret.append(wt)
    hd_ret.append(metric.hd(o, t))
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    tc = dice_score(o, t)
    ret.append(tc)
    hd_ret.append(metric.hd(o, t))
    # active
    o = (output == 3)
    t = (target == 3)
    et = dice_score(o, t)
    ret.append(et)
    hd_ret.append(metric.hd(o, t))

    return ret, hd_ret


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0
    t = target > 0  # ce
    wt = dice_score(o, t)
    ret.append(wt)
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    tc = dice_score(o, t)
    ret.append(tc)
    # active
    o = (output == 3)
    t = (target == 3)

    et = dice_score(o, t)
    ret.append(et)

    return ret


def get_separate_loss(output, target):
    target_1 = target.clone()
    target_2 = target.clone()
    target_4 = target.clone()
    # label 1
    label_1 = output['01']
    target_1[target_1 == 1] = 1
    target_1[target_1 == 2] = 0
    target_1[target_1 == 3] = 0
    alpha = 1.0
    out_target_1 = F.one_hot(target_1, 2)
    out_target_1 = out_target_1.permute(0, 4, 1, 2, 3).contiguous()

    dice = dice_loss(label_1, out_target_1, 2)
    loss = softmax_weighted_loss(label_1, out_target_1, 2)

    loss_01 = loss + dice * alpha

    # label 2
    label_2 = output['02']
    target_2[target_2 == 1] = 0
    target_2[target_2 == 2] = 1
    target_2[target_2 == 3] = 0
    out_target_2 = F.one_hot(target_2, 2)
    out_target_2 = out_target_2.permute(0, 4, 1, 2, 3).contiguous()

    # out_target_2[out_target_2 == 1] = 2

    dice_2 = dice_loss(label_2, out_target_2, 2)

    loss_2 = softmax_weighted_loss(label_2, out_target_2, 2)

    loss_02 = loss_2 + dice_2 * alpha

    # label 4
    label_4 = output['04']
    target_4[target_4 == 1] = 0
    target_4[target_4 == 2] = 0
    target_4[target_4 == 3] = 1
    out_target_4 = F.one_hot(target_4, 2)
    out_target_4 = out_target_4.permute(0, 4, 1, 2, 3).contiguous()

    # out_target_4[out_target_4 == 1] = 3

    dice_4 = dice_loss(label_4, out_target_4, 2)

    loss_4 = softmax_weighted_loss(label_4, out_target_4, 2)

    loss_04 = loss_4 + dice_4 * alpha

    return loss_01 + loss_02 + loss_04


def get_edge_separate_loss(output, target):
    target_1 = target.clone()
    target_2 = target.clone()
    target_4 = target.clone()

    alpha = 1.0

    label_1 = output['01']

    target_1[target == 5] = 1
    target_1[target == 6] = 1
    target_1[target == 7] = 1
    target_1[target == 2] = 0
    target_1[target == 4] = 0
    target_1[target == 8] = 0

    out_target_1 = F.one_hot(target_1, 2)
    out_target_1 = out_target_1.permute(0, 4, 1, 2, 3).contiguous()

    dice = dice_loss(label_1, out_target_1, 2)

    loss = softmax_weighted_loss(label_1, out_target_1, 2)

    loss_01 = loss + dice * alpha

    # label 2: 2 5 6 8
    label_2 = output['02']
    target_2[target == 2] = 1
    target_2[target == 5] = 1
    target_2[target == 6] = 1
    target_2[target == 8] = 1
    target_2[target == 1] = 0
    target_2[target == 4] = 0
    target_2[target == 7] = 0
    out_target_2 = F.one_hot(target_2, 2)
    out_target_2 = out_target_2.permute(0, 4, 1, 2, 3).contiguous()

    # out_target_2[out_target_2 == 1] = 2

    dice_2 = dice_loss(label_2, out_target_2, 2)

    loss_2 = softmax_weighted_loss(label_2, out_target_2, 2)

    loss_02 = loss_2 + dice_2 * alpha

    # label 4: 4 5 7 8
    label_4 = output['04']
    target_4[target == 4] = 1
    target_4[target == 5] = 1
    target_4[target == 7] = 1
    target_4[target == 8] = 1
    target_4[target == 1] = 0
    target_4[target == 2] = 0
    target_4[target == 6] = 0

    out_target_4 = F.one_hot(target_4, 2)
    out_target_4 = out_target_4.permute(0, 4, 1, 2, 3).contiguous()

    # out_target_4[out_target_4 == 1] = 3

    dice_4 = dice_loss(label_4, out_target_4, 2)

    loss_4 = softmax_weighted_loss(label_4, out_target_4, 2)

    loss_04 = loss_4 + dice_4 * alpha

    return loss_01 + loss_02 + loss_04

# def get_separate_loss(output, target):
#
#     target_1 = target.clone()
#     target_2 = target.clone()
#     target_4 = target.clone()
#     # label 1
#     label_1 = output['01']
#     target_1[target_1 == 1] = 1
#     target_1[target_1 == 2] = 0
#     target_1[target_1 == 3] = 0
#     alpha = 1.0
#
#     dice = dice_score(label_1[:, 1, ...], target_1.float())
#     loss = torch.nn.BCEWithLogitsLoss()(label_1[:, 1, ...], target_1.float())
#     loss_01 = loss + (1 - dice) * alpha
#
#     # label 2
#     label_2 = output['02']
#     target_2[target_2 == 1] = 0
#     target_2[target_2 == 2] = 2
#     target_2[target_2 == 3] = 0
#
#     dice_2 = dice_score(label_2[:, 1, ...], target_2.float())
#
#     loss_2 = torch.nn.BCEWithLogitsLoss()(label_2[:, 1, ...], target_2.float())
#
#     loss_02 = loss_2 + (1 - dice_2) * alpha
#
#     # label 4
#     label_4 = output['04']
#     target_4[target_4 == 1] = 0
#     target_4[target_4 == 2] = 0
#     target_4[target_4 == 3] = 3
#
#     dice_4 = dice_score(label_4[:, 1, ...], target_4.float())
#
#     loss_4 = torch.nn.BCEWithLogitsLoss()(label_4[:, 1, ...], target_4.float())
#
#     loss_04 = loss_4 + (1 - dice_4) * alpha
#
#     return loss_01 + loss_02 + loss_04
#
#
# def get_edge_separate_loss(output, target):
#     target_1 = target.clone()
#     target_2 = target.clone()
#     target_4 = target.clone()
#
#     alpha = 1.0
#
#     label_1 = output['01']
#
#     target_1[target_1 == 5] = 1
#     target_1[target_1 == 6] = 1
#     target_1[target_1 == 7] = 1
#     target_1[target_1 == 2] = 0
#     target_1[target_1 == 4] = 0
#     target_1[target_1 == 8] = 0
#
#     dice = dice_score(label_1[:, 1, ...], target_1.float())
#
#     loss = torch.nn.BCEWithLogitsLoss()(label_1[:, 1, ...], target_1.float())
#
#     loss_01 = loss + (1 - dice) * alpha
#
#     # label 2: 2 5 6 8
#     label_2 = output['02']
#     target_2[target_2 == 2] = 2
#     target_2[target_2 == 5] = 2
#     target_2[target_2 == 6] = 2
#     target_2[target_2 == 8] = 2
#     target_2[target_2 == 1] = 0
#     target_2[target_2 == 4] = 0
#     target_2[target_2 == 7] = 0
#
#     dice_2 = dice_score(label_2[:, 1, ...], target_2.float())
#
#     loss_2 = torch.nn.BCEWithLogitsLoss()(label_2[:, 1, ...], target_2.float())
#
#     loss_02 = loss_2 + (1 - dice_2) * alpha
#
#     # label 4: 4 5 7 8
#     label_4 = output['04']
#     target_4[target_4 == 4] = 3
#     target_4[target_4 == 5] = 3
#     target_4[target_4 == 7] = 3
#     target_4[target_4 == 8] = 3
#     target_4[target_4 == 1] = 0
#     target_4[target_4 == 2] = 0
#     target_4[target_4 == 6] = 0
#
#     dice_4 = dice_score(label_4[:, 1, ...], target_4.float())
#
#     loss_4 = torch.nn.BCEWithLogitsLoss()(label_4[:, 1, ...], target_4.float())
#
#     loss_04 = loss_4 + (1 - dice_4) * alpha
#
#     return loss_01 + loss_02 + loss_04
