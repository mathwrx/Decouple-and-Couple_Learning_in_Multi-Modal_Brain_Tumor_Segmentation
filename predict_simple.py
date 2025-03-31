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
import utils.hausdorff as hausdorff
import pandas as pd


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
        # utils.tools.get_seperate_loss(test_1[1], target)
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
    mIOU_score.append(mIOU(o=(output == 3), t=(target == 4)))
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


def softmax_output_mIou(output, target):
    ret = []

    # whole
    o = output > 0
    t = target > 0  # ce
    wt = mIOU(o, t)
    ret.append(wt)
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    tc = mIOU(o, t)
    ret.append(tc)
    # active
    o = (output == 3)
    t = (target == 3)
    et = mIOU(o, t)
    ret.append(et)

    return ret


def cal_hausdorff(output, target):
    ret = []
    # whole
    o = output > 0
    t = target > 0  # ce
    wt = hausdorff.hausdorff_distance_95(o, t)
    ret.append(wt)
    # core
    o = (output == 1) + (output == 3)
    t = (target == 1) + (target == 3)
    # if np.sum(o) < 500:
    #     o = o * 0
    tc = hausdorff.hausdorff_distance_95(o, t)
    ret.append(tc)
    # active
    o = (output == 3)
    t = (target == 3)
    # if np.sum(o) < 500:
    #     o = o * 0
    et = hausdorff.hausdorff_distance_95(o, t)
    ret.append(et)

    return ret


def exportSum(data, visual, modal):
    root = '/rundata'
    cvs_dir = os.path.join(root, '2018_sum.csv')
    if not os.path.exists(cvs_dir):
        log = pd.DataFrame(index=[], columns=[
            'name', 'wt', 'tc', 'et', 'sum', 'pre_1', 'pre_2', 'pre_4', 'gt_1', 'gt_2', 'gt_4',
        ])
        log.to_csv(cvs_dir, index=False)
    for i in range(20):
        item = data[i]
        tt_name = modal + "_" + item['name']
        a = {'name': [tt_name], 'wt': [item['wt']], 'tc': [item['tc']], 'et': [item['et']],
             'sum': [item['sum']], 'pre_1': [item['pre_1']], 'pre_2': [item['pre_2']],
             'pre_4': [item['pre_4']], 'gt_1': [item['gt_1']], 'gt_2': [item['gt_2']], 'gt_4': [item['gt_4']]}
        df = pd.DataFrame(a)
        df.to_csv(cvs_dir, mode='a', index=False, header=False)
    # mode = 'a'为追加数据，index为每行的索引序号，header为标题
    #
    for i in range(95):
        item = data[i]
        exportItemData(visual, item, modal)


def exportItemData(path, item, modal):
    cvs_dir = os.path.join(path, '2018_' + modal + '_sum.csv')
    if not os.path.exists(cvs_dir):
        log = pd.DataFrame(index=[], columns=[
            'name', 'wt', 'tc', 'et', 'sum', 'pre_1', 'pre_2', 'pre_4', 'gt_1', 'gt_2', 'gt_4',
        ])
        log.to_csv(cvs_dir, index=False)

    a = {'name': [item['name']], 'wt': [item['wt']], 'tc': [item['tc']], 'et': [item['et']],
         'sum': [item['sum']], 'pre_1': [item['pre_1']], 'pre_2': [item['pre_2']],
         'pre_4': [item['pre_4']], 'gt_1': [item['gt_1']], 'gt_2': [item['gt_2']], 'gt_4': [item['gt_4']]}
    df = pd.DataFrame(a)
    # mode = 'a'为追加数据，index为每行的索引序号，header为标题
    df.to_csv(cvs_dir, mode='a', index=False, header=False)


def output_pic(modal, visual, name, output, label):
    if not os.path.exists(os.path.join(visual, name)):
        os.makedirs(os.path.join(visual, name))

    predict_path = os.path.join(visual, name, 'predict')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    for frame in range(128):
        item = output[:, :, frame]
        Snapshot_img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
        Snapshot_img[:, :][np.where(item == 1)] = [250, 250, 149]  # [104, 147, 207]
        Snapshot_img[:, :][np.where(item == 2)] = [244, 130, 128]  # [242, 130, 129]
        Snapshot_img[:, :][np.where(item == 3)] = [97, 136, 200]
        imageio.imwrite(os.path.join(predict_path, modal + "_pre_" + str(frame) + '.png'), Snapshot_img)

    label_path = os.path.join(visual, name, 'label')
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    for frame in range(128):
        ll_item = label[:, :, frame]
        label_img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
        label_img[:, :][np.where(ll_item == 1)] = [250, 250, 149]  # [104, 147, 207]
        label_img[:, :][np.where(ll_item == 2)] = [244, 130, 128]  # [242, 130, 129]
        label_img[:, :][np.where(ll_item == 3)] = [97, 136, 200]
        imageio.imwrite(os.path.join(label_path, modal + '_label_' + str(frame) + '.png'), label_img)



def output_excel(modal, visual, name, output, label):

    if not os.path.exists(os.path.join(visual, name)):
        os.makedirs(os.path.join(visual, name))

    predict_path = os.path.join(visual, name, 'predict')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    name_path = os.path.join(predict_path, name)
    if not os.path.exists(name_path):
        os.makedirs(name_path)
    excel_all = []
    for frame in range(128):
        item = output[:, :, frame]
        label_item = label[:, :, frame]
        if label_item.max() > 0:
            dice = utils.tools.softmax_output_dice(item, label_item)
            item_name = name + "_" + str(frame)
            item_sum = {'name': item_name, 'wt': dice[0], 'tc': dice[1], 'et': dice[2],
                        'sum': dice[0] * dice[1] * dice[2]}
            excel_all.append(item_sum)
        # Snapshot_img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
        # Snapshot_img[:, :][np.where(item == 1)] = [250, 250, 149]  # [104, 147, 207]
        # Snapshot_img[:, :][np.where(item == 2)] = [244, 130, 128]  # [242, 130, 129]
        # Snapshot_img[:, :][np.where(item == 3)] = [97, 136, 200]
        # imageio.imwrite(os.path.join(predict_path, modal + "_pre_" + str(frame) + '.png'), Snapshot_img)
    len_e = len(excel_all)
    excel_all.sort(key=lambda x: x['wt'])
    export_item_excel(modal, name_path, name, '_wt', excel_all, len_e)

    excel_all.sort(key=lambda x: x['tc'])
    export_item_excel(modal, name_path, name, '_tc', excel_all, len_e)

    excel_all.sort(key=lambda x: x['et'])
    export_item_excel(modal, name_path, name, '_et', excel_all, len_e)
    # cvs_dir_tc = os.path.join(name_path, name + '_tc' + '.csv')
    # cvs_dir_et = os.path.join(name_path, name + '_et' + '.csv')
    # if not os.path.exists(cvs_dir_wt):
    #     log = pd.DataFrame(index=[], columns=[
    #         'name', 'wt', 'tc', 'et', 'sum'
    #     ])
    #     log.to_csv(cvs_dir_wt, index=False)
    # for le in range(len_e):
    #     le_item = excel_all[le]
    #     a = {'name': [le_item['name']], 'wt': [le_item['wt']], 'tc': [le_item['tc']], 'et': [le_item['et']],
    #          'sum': [le_item['sum']]}
    #     df = pd.DataFrame(a)
    #     # mode = 'a'为追加数据，index为每行的索引序号，header为标题
    #     df.to_csv(cvs_dir_wt, mode='a', index=False, header=False)


def export_item_excel(modal, name_path, name, region, excel_all, len_e):
    cvs_dir = os.path.join(name_path, modal + '_' + name + region + '.csv')
    if not os.path.exists(cvs_dir):
        log = pd.DataFrame(index=[], columns=[
            'name', 'wt', 'tc', 'et', 'sum'
        ])
        log.to_csv(cvs_dir, index=False)
    for le in range(len_e):
        le_item = excel_all[le]
        a = {'name': [le_item['name']], 'wt': [le_item['wt']], 'tc': [le_item['tc']], 'et': [le_item['et']],
             'sum': [le_item['sum']]}
        df = pd.DataFrame(a)
        # mode = 'a'为追加数据，index为每行的索引序号，header为标题
        df.to_csv(cvs_dir, mode='a', index=False, header=False)

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
    data_name = '2018_excel'
    if not os.path.exists(os.path.join(visual, data_name)):
        os.makedirs(os.path.join(visual, data_name))
    visual = os.path.join(visual, data_name)
    H, W, T = 240, 240, 160
    model.eval()

    runtimes = []
    wt_dices = []
    tc_dices = []
    et_dices = []

    haus_wt_dices = []
    haus_tc_dices = []
    haus_et_dices = []

    miou_wt_dices = []
    miou_tc_dices = []
    miou_et_dices = []

    sum_sort = []
    print('sum=====', sum(x.numel() for x in model.parameters()))
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            # data = [t.cuda(non_blocking=True) for t in data]
            x, target, edge, missing_modal, path = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            edge = edge.cuda(non_blocking=True)
        else:
            x, missing_modal = data
            x.cuda()
        # outputs = model(x, missing_modal)
        logit = F.softmax(model(x, missing_modal)[0], 1)  # no flip
        logit += F.softmax(model(x.flip(dims=(2,)), missing_modal)[0].flip(dims=(2,)),
                           1)  # flip W, D
        logit += F.softmax(model(x.flip(dims=(3,)), missing_modal)[0].flip(dims=(3,)),
                           1)  # flip W, D
        logit += F.softmax(model(x.flip(dims=(4,)), missing_modal)[0].flip(dims=(4,)),
                           1)  # flip W, D
        logit += F.softmax(model(x.flip(dims=(2, 3)), missing_modal)[0].flip(dims=(2, 3)),
                           1)  # flip W, D
        logit += F.softmax(model(x.flip(dims=(2, 4)), missing_modal)[0].flip(dims=(2, 4)),
                           1)  # flip W, D
        logit += F.softmax(model(x.flip(dims=(3, 4)), missing_modal)[0].flip(dims=(3, 4)),
                           1)  # flip W, D
        logit += F.softmax(model(x.flip(dims=(2, 3, 4)), missing_modal)[0].flip(dims=(2, 3, 4)),
                           1)  # flip H, W, D

        logit = logit / 8
        # s_loss = utils.tools.get_separate_loss(outputs[1], target)
        # outputs = tailor_and_concat(x, missing_modal, model, None)
        # output = outputs[0, :, :H, :W, : T].cpu().detach().numpy()

        # output = outputs[0].cpu().detach().numpy()
        output = logit.cpu().detach().numpy()
        output = output.argmax(1)
        #
        # image = nib.load(path[0] + 'seg.nii.gz')
        # filepath = os.path.join(savepath, '{}.nii.gz'.format(names[i]))
        # new_nift = nib.Nifti1Image(output[0, ...], None)
        # nib.save(new_nift, filepath)

        # Snapshot_img = np.zeros(shape=(128, 128, 3, 128), dtype=np.uint8)
        # Snapshot_img[:, :][np.where(output[0, ...] == 1)] = [250, 250, 149]  # [104, 147, 207]
        # Snapshot_img[:, :][np.where(output[0, ...] == 2)] = [244, 130, 128]  # [242, 130, 129]
        # Snapshot_img[:, :][np.where(output[0, ...] == 3)] = [97, 136, 200]
        # predict_path = os.path.join(visual, names[i], 'predict')
        # if not os.path.exists(predict_path):
        #     os.makedirs(predict_path)
        # for frame in range(128):
        #     if not os.path.exists(os.path.join(predict_path, names[i])):
        #         os.makedirs(os.path.join(predict_path, names[i]))
        #     imageio.imwrite(os.path.join(predict_path, names[i], str(frame) + '.png'), Snapshot_img[:, :, :, frame])
        # file_names = ['supervise_loss', 'edge_loss', 'mid_semantic_loss', 'mid_edge_loss']
        # for u_index in range(4):
        #     u_name = file_names[u_index]
        #     u_data = outputs[u_index + 1]['01'].cpu().detach().numpy()
        #     u_data = u_data.argmax(1)
        #     Snapshot_img = np.zeros(shape=(128, 128, 3, 128), dtype=np.uint8)
        #     Snapshot_img[:, :, 0, :][np.where(u_data[0] == 1)] = 255
        #     for frame in range(128):
        #         if not os.path.exists(os.path.join(visual, names[i], u_name)):
        #             os.makedirs(os.path.join(visual, names[i], u_name))
        #         imageio.imwrite(os.path.join(visual, names[i], u_name, str(frame) + '.png'), Snapshot_img[:, :, :, frame])
        name = names[i]
        num_0 = np.sum(output == 0)
        num_1 = np.sum(output == 1)
        num_2 = np.sum(output == 2)
        num_3 = np.sum(output == 3)
        num_4 = output.max()
        # 计算dice
        # target_155 = target[0, :, :, :155].cpu().detach().numpy()
        target_155 = target.cpu().detach().numpy()
        #output_pic('Our', visual, name, output[0, ...], target_155[0, ...])
        soft = utils.tools.softmax_output_dice(output, target_155)
        miou = softmax_output_mIou(output, target_155)
        output_excel('Our', visual, name, output[0, ...], target_155[0, ...])
        # item_sum = {'name': name, 'wt': soft[0], 'tc': soft[1], 'et': soft[2], 'sum': soft[0] * soft[1] * soft[2],
        #             'pre_1': num_1, 'pre_2': num_2, 'pre_4': num_3, 'gt_1': np.sum(target_155 == 1),
        #             'gt_2': np.sum(target_155 == 2), 'gt_4': np.sum(target_155 == 3)}

        #sum_sort.append(item_sum)
        print('0标签:{},1标签:{},2标签:{},3标签:{},索引最大值: {}'.format(num_0, num_1, num_2, num_3, num_4))
        print('GT: 0标签:{},1标签:{},2标签:{},3标签:{},索引最大值: {}'.format(np.sum(target_155 == 0), np.sum(target_155 == 1),
                                                                 np.sum(target_155 == 2), np.sum(target_155 == 3),
                                                                 target_155.max()))
        haus = [0, 0, 0]
        haus = cal_hausdorff(output, target_155)
        print('name:{}, msg={}, DICE= WT:{},TC:{},ET:{}'.format(name, msg, soft[0], soft[1], soft[2]))
        print('name:{}, msg={}, MIOU= WT:{},TC:{},ET:{}'.format(name, msg, miou[0], miou[1], miou[2]))
        print('name:{}, msg={}, HAUSDORFF= WT:{},TC:{},ET:{}'.format(name, msg, haus[0], haus[1], haus[2]))

        wt_dices.append(soft[0])

        tc_dices.append(soft[1])

        et_dices.append(soft[2])

        haus_wt_dices.append(haus[0])
        haus_tc_dices.append(haus[1])
        haus_et_dices.append(haus[2])

        miou_wt_dices.append(miou[0])
        miou_tc_dices.append(miou[1])
        miou_et_dices.append(miou[2])
    # sum_sort.sort(key=lambda x: x['et'])
    # exportSum(sum_sort, visual, 'Our')
    print('WT Dice: %.4f' % np.mean(wt_dices))

    print('TC Dice: %.4f' % np.mean(tc_dices))

    print('ET Dice: %.4f' % np.mean(et_dices))

    print('HAUSDORFF DIS WT: %.4f' % np.mean(haus_wt_dices))

    print('HAUSDORFF DIS TC: %.4f' % np.mean(haus_tc_dices))

    print('HAUSDORFF DIS ET: %.4f' % np.mean(haus_et_dices))

    print('MIOU  WT : %.4f' % np.mean(miou_wt_dices))

    print('MIOU  TC : %.4f' % np.mean(miou_tc_dices))

    print('MIOU  ET : %.4f' % np.mean(miou_et_dices))

    # print('runtimes:', sum(runtimes)/len(runtimes))

    return np.mean(wt_dices), np.mean(tc_dices), np.mean(et_dices)
