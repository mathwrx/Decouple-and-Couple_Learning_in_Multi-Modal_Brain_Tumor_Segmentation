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
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

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
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            # data = [t.cuda(non_blocking=True) for t in data]
            x, target, edge, missing_modal = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        else:
            x, missing_modal = data
            x.cuda()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit = tailor_and_concat(x, missing_modal, model)

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            runtimes.append(elapsed_time)


            if multimodel:
                logit = F.softmax(logit, dim=1)
                output = logit / 4.0

                load_file1 = load_file.replace('7998', '7996')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7997')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7999')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
            else:
                output = F.softmax(logit, dim=1)



        else:

            x = x[..., :155]

            logit = F.softmax(tailor_and_concat(x, missing_modal, model,target), 1)  # no flip

            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), missing_modal, model).flip(dims=(2,)), 1)  # flip H

            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), missing_modal, model).flip(dims=(3,)), 1)  # flip W

            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), missing_modal, model).flip(dims=(4,)), 1)  # flip D

            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), missing_modal, model).flip(dims=(2, 3)),
                               1)  # flip H, W

            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), missing_modal, model).flip(dims=(2, 4)),
                               1)  # flip H, D

            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), missing_modal, model).flip(dims=(3, 4)),
                               1)  # flip W, D

            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), missing_modal, model).flip(dims=(2, 3, 4)),
                               1)  # flip H, W, D
            output = logit / 8.0  # mean

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()

        output = output.argmax(0)

        target_155 = target[0, ...].cpu().detach().numpy()

        target_155 = target_155[..., :155]

        soft = softmax_output_dice(output, target_155)

        print(msg, soft)

        wt_dices.append(soft[0])

        tc_dices.append(soft[1])

        et_dices.append(soft[2])

    print('WT Dice: %.4f' % np.mean(wt_dices))

    print('TC Dice: %.4f' % np.mean(tc_dices))

    print('ET Dice: %.4f' % np.mean(et_dices))

    # print('runtimes:', sum(runtimes)/len(runtimes))
