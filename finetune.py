from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='kitti2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--threshold', type=int, default=3)
parser.add_argument('--within_max_disp', action='store_true')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == 'kitti2015':
    from dataloader import KITTIloader2015 as ls
    from dataloader import KITTILoader as DA
elif args.datatype == 'kitti2012':
    from dataloader import KITTIloader2012 as ls
    from dataloader import KITTILoader as DA
elif args.datatype == 'middleburry':
    from dataloader import Middlebury as ls
    from dataloader import Middlebury as DA
elif args.datatype == 'scared':
    from dataloader import Scared as ls
    from dataloader import Scared as DA

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    if args.within_max_disp:
        mask = torch.logical_and(disp_true > 0.0, disp_true < args.maxdisp)
    else:
        mask = disp_true > 0.0
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output3, 1)
        loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data[0]


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()

    # NOTE: there is a bug reported in the repo, disparity needs to be scaled by 1.17
    pred_disp = pred_disp * 1.17

    mask = disp_true > 0.0
    error = F.l1_loss(pred_disp[mask], disp_true[mask], reduction='none')
    correct = torch.sum(error <= args.threshold)
    total = error.numel()

    torch.cuda.empty_cache()

    # import matplotlib.pyplot as plt
    # for i in range(imgL.size(0)):
    #     plt.figure()
    #     plt.imshow(disp_true[i])
    #     plt.figure()
    #     plt.imshow(pred_disp[i])
    #     plt.show()

    return float(correct), float(total), torch.mean(error)


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    ## Test ##
    total_correct = 0
    total_px = 0
    avg_error = 0.0
    epoch = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        correct, total, epe = test(imgL, imgR, disp_L)
        print('Iter %d 3-px error in val = %.3f' % (batch_idx, 100 - correct / total * 100))
        total_correct = total_correct + correct
        total_px = total_px + total
        avg_error += epe

    total_test_loss = 100 - total_correct / total_px * 100
    print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss))
    print('epoch %d total epe error in val = %.3f' % (epoch, avg_error / len(TestImgLoader)))

    # if total_test_loss > max_acc:
    #     max_acc = total_test_loss
    #     max_epo = epoch
    # print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

    # # SAVE
    # savefilename = args.savemodel + 'finetune_' + str(epoch) + '.tar'
    # torch.save({
    #     'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #     'test_loss': total_test_loss,
    # }, savefilename)
    #
    # print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    # print(max_epo)
    # print(max_acc)


if __name__ == '__main__':
    main()
