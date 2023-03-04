import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# 训练中采用了动态调整学习率的策略，在第10，12，14轮训练的时候，让learning_rate除以2变为更小的学习率
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
#  weight decay策略，作为Adam优化器超参数，实现中并未使用
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
# 深度假设数量，一共假设这么多种不同的深度，在里面找某个像素的最优深度
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
# 深度假设间隔缩放因子，每隔interval假设一个新的深度，这个interval要乘以这个scale
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
# loadckpt, logdir, resume: 主要用来控制从上次学习中恢复继续训练的参数
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
# 'store_true’表示终端运行时（action）时为真，否则为假
# ’store_false’表示触发时（action）时为假
parser.add_argument('--resume', action='store_true', help='continue to train the model')
# 输出到tensorboard中的信息频率
parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
# 保存模型频率，默认是训练一整个epoch保存一次模型
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

model = MVSNet(refine=False)

with torch.no_grad():
    imgs = torch.rand((4, 3, 3, 512, 640)).cuda()
    proj_matrices = torch.rand((4, 3, 4, 4)).cuda()
    depth_values = torch.rand((4, 192)).cuda()
    model(imgs, proj_matrices, depth_values)
