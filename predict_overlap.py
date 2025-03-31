import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils.tools

cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio



def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def tailor_and_concat(x, missing_modal, model, target=None):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()

    for i in range(len(temp)):
        test_1 = model(temp[i], missing_modal)
        temp[i] = test_1[0]
        #utils.tools.get_seperate_loss(test_1[1], target)
    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0
    t = target > 0  # ce
    ret += dice_score(o, t)
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += dice_score(o, t)
    # active
    o = (output == 3)
    t = (target == 3)
    ret += dice_score(o, t)

    return ret


keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        load_file,
        multimodel,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False,  # if you are valid when train
        ):

    H, W, T = 240, 240, 160
    model.eval()

    runtimes = []
    wt_dices = []
    tc_dices = []
    et_dices = []
    print('sum=====', sum(x.numel() for x in model.parameters()))
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            # data = [t.cuda(non_blocking=True) for t in data]
            x, target, edge, missing_modal = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            edge = edge.cuda(non_blocking=True)
        else:
            x, missing_modal = data
            x.cuda()
        #outputs = model(x, missing_modal)
        outputs = tailor_and_concat(x, missing_modal, model, None)
        output = outputs[0, :, :H, :W, : T].cpu().detach().numpy()
        #output = outputs[0].cpu().detach().numpy()
        output = output.argmax(0)
        num_0 = np.sum(output == 0)
        num_1 = np.sum(output == 1)
        num_2 = np.sum(output == 2)
        num_3 = np.sum(output == 3)
        num_4 = output.max()
        # 计算dice
        target_155 = target[0, :, :, :155].cpu().detach().numpy()
        #target_155 = target.cpu().detach().numpy()
        target_155[target_155 == 4] = 3
        soft = utils.tools.softmax_output_dice(output, target_155)
        name = names[i]
        print('name:{}, msg={}, DICE= WT:{},TC:{},ET:{}'.format(name, msg, soft[0], soft[1], soft[2]))
        print('0标签:{},1标签:{},2标签:{},3标签:{},索引最大值: {}'.format(num_0, num_1, num_2, num_3, num_4))
        wt_dices.append(soft[0])

        tc_dices.append(soft[1])

        et_dices.append(soft[2])

    print('WT Dice: %.4f' % np.mean(wt_dices))

    print('TC Dice: %.4f' % np.mean(tc_dices))

    print('ET Dice: %.4f' % np.mean(et_dices))

    # print('runtimes:', sum(runtimes)/len(runtimes))
    
    return np.mean(wt_dices), np.mean(tc_dices), np.mean(et_dices)