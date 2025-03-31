import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.ClsWiseBraTS128Test import BraDataSet128
from predict_overlap import validate_softmax
from models.clswiseformer.cls_wise_former import get_cls_wise_former

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='bitgroup_down_128', type=str)

parser.add_argument('--project_root', default='cls_wise_attention_one_3_to_4', type=str)

parser.add_argument('--root', default='2-MICCAI_BraTS_2018', type=str)

parser.add_argument('--valid_dir', default='MICCAI_BraTS_2018_Data_Training', type=str)

parser.add_argument('--valid_file', default='train.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission2', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='clswiseformerone_trans', type=str)

parser.add_argument('--test_date', default='20220731', type=str)

parser.add_argument('--test_file', default='model_epoch_649.pth', type=str)

parser.add_argument('--use_TTA', default=True, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=0, type=int)

# parser.add_argument('--model_name', default='mmformer', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    model = get_cls_wise_former(dataset='brats', _conv_repr=True, _pe_type="fixed", gpu=args.gpu)

    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    #model = model.cuda(args.gpu)
    load_file = os.path.join(args.project_root,
                             'checkpoint', args.experiment+args.test_date, args.test_file)

    print('load file ----------', load_file)
    if os.path.exists(load_file):
        checkpoint = torch.load(load_file, map_location={'cuda:1': 'cuda:0'})
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraDataSet128(valid_list, valid_root, mode='valid', drop_modal=False)
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(args.project_root, args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(args.project_root, args.output_dir,
                          args.visual, args.experiment+args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True,
                         valid_in_train=True,
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    print('start test -----------------------------------------------')
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0,1,2,3,4,5')
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


