# python3 -m torch.distributed.launch --nproc_per_node=3 --master_port 20004 train.py

import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim

import utils.tools
from models.clswiseformer.cls_wise_former import get_cls_wise_former
import torch.distributed as dist
from models import criterions

from data.ClsWiseBraTS128 import BraDataSet128
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from torch import nn
from torch.cuda.amp import GradScaler, autocast

local_time = time.strftime("%Y%m%d %H%M%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='bitgroup_mul', type=str)

parser.add_argument('--experiment', default='clswiseformer_mul', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='cls_wise,'
                            'training on train.txt!',
                    type=str)

parser.add_argument('--project_root', default='cls_wise_mul', type=str)

# DataSet Information
parser.add_argument('--root', default='2-MICCAI_BraTS_2018', type=str)

parser.add_argument('--train_dir', default='MICCAI_BraTS_2018_Data_Training', type=str)

parser.add_argument('--valid_dir', default='Valid', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--dataset', default='brats', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=160, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1,2,3,4,5', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=1000, type=int)

parser.add_argument('--save_freq', default=50, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--print_rank', default=0, type=int, help='node rank for distributed training')
args = parser.parse_args()


def main_worker():
    print('args.local_rank==================', args.local_rank, args.load, args.resume)
    if args.local_rank == args.print_rank:
        log_dir = os.path.join(args.project_root, 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        # if not os.path.isfile(log_file):

        log_args(log_file)
        # logging.info('--------------------------------------This is all argsurations----------------------------------')
        # for arg in vars(args):
        #     logging.info('{}={}'.format(arg, getattr(args, arg)))
        # logging.info('----------------------------------------This is a halving line----------------------------------')
        # logging.info('{}'.format(args.description))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    model = get_cls_wise_former(dataset='brats', _conv_repr=True, _pe_type="fixed", gpu=args.local_rank)

    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = getattr(criterions, args.criterion)

    if args.local_rank == args.print_rank:
        checkpoint_dir = os.path.join(args.project_root, 'checkpoint', args.experiment + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    resume = args.resume

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    train_set_full_modal = BraDataSet128(train_list, train_root, args.mode, drop_modal=False)
    train_sampler_full_modal = torch.utils.data.distributed.DistributedSampler(train_set_full_modal)

    num_gpu = 4  # (len(args.gpu)+1) // 2

    train_loader_full_modal = DataLoader(dataset=train_set_full_modal, sampler=train_sampler_full_modal,
                                         batch_size=args.batch_size,
                                         drop_last=False, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()
    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        logging.info('current proc title==== {}'.format(setproctitle.getproctitle()))
        train_sampler_full_modal.set_epoch(epoch)  # shuffle
        # train_sampler_missing_modal.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()

        train_loader = train_loader_full_modal
        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target, edge, missing_modal = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            edge = edge.cuda(args.local_rank, non_blocking=True)
            # output = model(x, missing_modal)
            outputs = model(x, missing_modal)
            if args.local_rank == args.print_rank:
                cal_out = outputs[0][0, :, :, :, :].cpu().detach().numpy()
                cal_out = cal_out.argmax(0)
                num_0 = np.sum(cal_out == 0)
                num_1 = np.sum(cal_out == 1)
                num_2 = np.sum(cal_out == 2)
                num_3 = np.sum(cal_out == 3)
                num_4 = cal_out.max()
                # 计算dice
                target_155 = target.cpu().detach().numpy()
                dice = utils.tools.softmax_output_dice(cal_out, target_155)
                logging.info('epoch:{}, DICE= WT:{},TC:{},ET:{}'.format(epoch, dice[0], dice[1], dice[2]))
                logging.info('epoch:{}, 0标签:{},1标签:{},2标签:{},3标签:{},索引最大值: {}'
                             .format(epoch, num_0, num_1, num_2, num_3, num_4))
            loss = criterion(outputs[0], target)
            p_out = loss.item()
            s_loss = utils.tools.get_separate_loss(outputs[1], target)
            edge_loss = utils.tools.get_edge_separate_loss(outputs[2], edge)
            mid_s_loss = utils.tools.get_separate_loss(outputs[3], target)
            mid_edge_loss = utils.tools.get_edge_separate_loss(outputs[4], edge)
            loss = loss + s_loss + edge_loss + mid_s_loss + mid_edge_loss

            #loss = loss + mid_s_loss + mid_edge_loss
           # loss = loss + s_loss + edge_loss
            #loss = loss + s_loss
            dist.barrier()
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            # reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
            # reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
            # reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()
            reduce_s_loss = all_reduce_tensor(s_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_edge_loss = all_reduce_tensor(edge_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_mid_s_loss = all_reduce_tensor(mid_s_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_mid_edge_loss = all_reduce_tensor(mid_edge_loss, world_size=num_gpu).data.cpu().numpy()
            if args.local_rank == args.print_rank:
                # logging.info(
                #     'Epoch All_reduce: {}_Iter:{}  loss: {:.5f} || s_loss:{:.4f} || edge_loss:{:.4f} || mid_s_loss:{:.4f} || mid_edge_loss:{:.4f} ||'
                #     .format(epoch, i, loss, 0, 0, 0, 0))
                logging.info('Epoch All_reduce: {}_Iter:{}  loss: {:.5f} || end_loss: {:.5f} || s_loss:{:.4f} || edge_loss:{:.4f} || mid_s_loss:{:.4f} || mid_edge_loss:{:.4f} ||'
                             .format(epoch, i, loss, p_out, s_loss, edge_loss, mid_s_loss, mid_edge_loss))

            optimizer.zero_grad()
            loss.backward()
            # for name,param in model.named_parameters():
            #     # if name.find('conv_label_2') > 0:
            #     #     print(name, param.grad)
            #     if param.grad is None:
            #         print(name)
            optimizer.step()

        end_epoch = time.time()
        if args.local_rank == args.print_rank:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

    if args.local_rank == args.print_rank:
        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')



def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):

    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epoch, power), 8)


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    new_gpu = [2, 3, 4, 5]
    args.print_rank = new_gpu[0]
    args.local_rank = new_gpu[args.local_rank]
    main_worker()
