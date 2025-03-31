import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
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

def calNum(output):
    cal_out = output[0, :, :, :, :].cpu().detach().numpy()
    cal_out = cal_out.argmax(0)
    num_0 = np.sum(cal_out == 0)
    num_1 = np.sum(cal_out == 1)
    num_2 = np.sum(cal_out == 2)
    num_3 = np.sum(cal_out == 3)
    num_4 = cal_out.max()
    print('网络返回值 0,1,2,3,索引最大值--------------------------------', num_0, num_1, num_2, num_3, num_4)

def tailor_and_concat(x, missing_modal, model):
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
        pre = model(temp[i], missing_modal)[0]
        calNum(pre)
        temp[i] = pre

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
    o = (output == 1) | (output == 4)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 4);t = (target == 4)
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
    ET_voxels_pred_list = []
    wt_dices = []
    tc_dices = []
    et_dices = []
    low_0_1 = []
    for i, data in enumerate(valid_loader):
        #print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            # data = [t.cuda(non_blocking=True) for t in data]
            x, target, missing_modal = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        else:
            x, missing_modal = data
            x.cuda()
        #print('===================================', x.shape)
        out = model(x, missing_modal)[0]
        output = out[0, :, :128, :128, :128].cpu().detach().numpy()
        output = output.argmax(0)
        #
        Snapshot_img = np.zeros(shape=(128, 128, 128), dtype=np.uint8)
        Snapshot_img[output == 0] = 0
        Snapshot_img[output == 1] = 1
        Snapshot_img[output == 2] = 2
        Snapshot_img[output == 3] = 4
        num_0 = np.sum(Snapshot_img == 0)
        num_1 = np.sum(Snapshot_img == 1)
        num_2 = np.sum(Snapshot_img == 2)
        num_4 = np.sum(Snapshot_img == 4)
        #print('标签0,1,2,4--------------------------------', num_0, num_1, num_2, num_4)
        target_128 = target[0, :, :, :].cpu().detach().numpy()
        soft = softmax_output_dice(Snapshot_img, target_128)
        wt_dices.append(soft[0])
        tc_dices.append(soft[1])
        et_dices.append(soft[2])
        #print('插值--------------------------------', names[i], i)
        #print(soft)
        #x = x[..., :155]
        #output = tailor_and_concat(x, missing_modal, model)
        #output = output[0, :, :, :, :].cpu().detach().numpy()
        #output = output.argmax(0)
        #target_155 = target[0, :, :, :].cpu().detach().numpy()
        #target_155 = target_155[..., :155]
        #soft = softmax_output_dice(output, target_155)
        #target_155[target_155 == 4] = 3
        # output = out[0, :, :128, :128, :128].cpu().detach().numpy()
        # output = output.argmax(0)
        #
        # Snapshot_img = np.zeros(shape=(128, 128, 128), dtype=np.uint8)
        # Snapshot_img[output == 1] = 1
        # Snapshot_img[output == 2] = 2
        # Snapshot_img[output == 3] = 4
        # num_0 = np.sum(Snapshot_img == 0)
        # num_1 = np.sum(Snapshot_img == 1)
        # num_2 = np.sum(Snapshot_img == 2)
        # num_4 = np.sum(Snapshot_img == 4)
        # print('标签0,1,2,4--------------------------------', num_0, num_1, num_2, num_4)
        # target_128 = target[0, :, :, :].cpu().detach().numpy()
        # soft = softmax_output_dice(Snapshot_img, target_128)
        #wt_dices.append(soft[0])
        #tc_dices.append(soft[1])
        #et_dices.append(soft[2])
        #print('插值--------------------------------', names[i], i)
        #print(soft)
        # print('插值--------------------------------', names[i], i)
        # calNum(out)

        # num_0 = np.sum(cal_out == 0)
        # num_1 = np.sum(cal_out == 1)
        # num_2 = np.sum(cal_out == 2)
        # num_4 = np.sum(cal_out == 4)
        # print('标签0,1,2,4--------------------------------', num_0, num_1, num_2, num_4)
        # if not use_TTA:
        #     torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        #     start_time = time.time()
        #     logit = tailor_and_concat(x, missing_modal, model)
        #
        #     torch.cuda.synchronize()
        #     elapsed_time = time.time() - start_time
        #     logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
        #     runtimes.append(elapsed_time)
        #
        #
        #     if multimodel:
        #         logit = F.softmax(logit, dim=1)
        #         output = logit / 4.0
        #
        #         load_file1 = load_file.replace('7998', '7996')
        #         if os.path.isfile(load_file1):
        #             checkpoint = torch.load(load_file1)
        #             model.load_state_dict(checkpoint['state_dict'])
        #             print('Successfully load checkpoint {}'.format(load_file1))
        #             logit = tailor_and_concat(x, model)
        #             logit = F.softmax(logit, dim=1)
        #             output += logit / 4.0
        #         load_file1 = load_file.replace('7998', '7997')
        #         if os.path.isfile(load_file1):
        #             checkpoint = torch.load(load_file1)
        #             model.load_state_dict(checkpoint['state_dict'])
        #             print('Successfully load checkpoint {}'.format(load_file1))
        #             logit = tailor_and_concat(x, model)
        #             logit = F.softmax(logit, dim=1)
        #             output += logit / 4.0
        #         load_file1 = load_file.replace('7998', '7999')
        #         if os.path.isfile(load_file1):
        #             checkpoint = torch.load(load_file1)
        #             model.load_state_dict(checkpoint['state_dict'])
        #             print('Successfully load checkpoint {}'.format(load_file1))
        #             logit = tailor_and_concat(x, model)
        #             logit = F.softmax(logit, dim=1)
        #             output += logit / 4.0
        #     else:
        #         output = F.softmax(logit, dim=1)
        #
        #
        # else:
        #     x = x[..., :155]
        #     #output = tailor_and_concat(x, missing_modal, model)
        #     logit = F.softmax(tailor_and_concat(x, missing_modal, model), 1)  # no flip
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), missing_modal, model).flip(dims=(2,)), 1)  # flip H
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), missing_modal, model).flip(dims=(3,)), 1)  # flip W
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), missing_modal, model).flip(dims=(4,)), 1)  # flip D
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), missing_modal, model).flip(dims=(2, 3)), 1)  # flip H, W
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), missing_modal, model).flip(dims=(2, 4)), 1)  # flip H, D
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), missing_modal, model).flip(dims=(3, 4)), 1)  # flip W, D
        #     logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), missing_modal, model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
        #     output = logit / 5.0  # mean
        #
        # output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        # output = output.argmax(0)
        # print('输出最大值-------------------', output.max())
        # seg_new = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]), dtype=np.uint8)
        # target_155 = target[0, ...].cpu().detach().numpy()
        # target_155 = target_155[..., :155]
        #
        # for index in range(output.shape[2]):
        #     for x_index in range(output.shape[0]):
        #         for y_index in range(output.shape[1]):
        #                 if output[x_index,y_index, index] == 0:
        #                     seg_new[x_index,y_index, index] = 1
        #                 if output[x_index,y_index, index] == 1:
        #                     seg_new[x_index,y_index, index] = 0
        #                 if output[x_index,y_index, index] == 2:
        #                     seg_new[x_index,y_index, index] = 2
        #                 if output[x_index,y_index, index] == 3:
        #                     seg_new[x_index,y_index, index] = 4
        # soft = softmax_output_dice(seg_new, target_155)
        # wt_dices.append(soft[0])
        # tc_dices.append(soft[1])
        # if soft[1] < 0.1:
        #     low_0_1.append(names[i])
        # et_dices.append(soft[2])
        # print(soft)
        # num_0 = np.sum(output == 0)
        # new_0 = np.sum(seg_new == 0)
        # num_1 = np.sum(output == 1)
        # new_1 = np.sum(seg_new == 1)
        # num_2 = np.sum(output == 2)
        # new_2 = np.sum(seg_new == 2)
        # num_3 = np.sum(output == 3)
        # num_4 = np.sum(output == 4)
        # new_4 = np.sum(seg_new == 4)
        #
        # if new_0 != num_1:
        #     print('填充错误--------------------')
        # C_SUM = 240 * 240 * 155
        # if num_4 > 0:
        #     print('有4个分类的代码--------------------')
        # if num_1 > C_SUM:
        #     print('有问题--------------------------------')
        # else:
        #     print('0,1,2,3,4--------------------------------', num_0,num_1,num_2,num_3,num_4)
        #     print('new 0,1,2,3,4--------------------------------', new_0, new_1, new_2, new_4)
        #     print('插值--------------------------------', names[i], i, C_SUM, num_1, C_SUM-num_1)
        # name = str(i)
        # if names:
        #     name = names[i]
        #     msg += '{:>20}, '.format(name)
        #
        # print(msg)
        #
        # if savepath:
        #     # .npy for further model ensemble
        #     # .nii for directly model submission
        #     assert save_format in ['npy', 'nii']
        #     if save_format == 'npy':
        #         np.save(os.path.join(savepath, name + '_preds'), output)
        #     if save_format == 'nii':
        #         # raise NotImplementedError
        #         oname = os.path.join(savepath, name + '.nii.gz')
        #         seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)
        #
        #         seg_img[np.where(output == 1)] = 1
        #         seg_img[np.where(output == 2)] = 2
        #         seg_img[np.where(output == 3)] = 4
        #         if verbose:
        #             print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
        #             print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
        #                   np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
        #         #nib.save(nib.Nifti1Image(seg_img, None), oname)
        #         #print('Successfully save {}'.format(oname))
        #
        #         if snapshot:
        #             """ --- grey figure---"""
        #             # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
        #             # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
        #             # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
        #             # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
        #             """ --- colorful figure--- """
        #             Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
        #             if output.min() < 1:
        #                 print('output ----------------------    00000000000000000')
        #             Snapshot_img[:, :, 0, :][np.where(output == 0)] = 255
        #             Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
        #             Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255
        #
        #             for frame in range(T):
        #                 if not os.path.exists(os.path.join(visual, name)):
        #                     os.makedirs(os.path.join(visual, name))
        #                 # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
        #                 imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])

    print('WT Dice: %.4f' % np.mean(wt_dices))
    print('TC Dice: %.4f' % np.mean(tc_dices))
    print('ET Dice: %.4f' % np.mean(et_dices))
    return np.mean(wt_dices), np.mean(tc_dices), np.mean(et_dices)
   # print(low_0_1)
    # print('runtimes:', sum(runtimes)/len(runtimes))
