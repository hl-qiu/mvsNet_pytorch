import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
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
# weight decay策略，作为Adam优化器超参数，实现中并未使用
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
# 深度假设数量，一共假设这么多种不同的深度，在里面找某个像素的最优深度，默认是192
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

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath
# 为CPU设置种子用于生成随机数
# 这样的意义在于可以保证在深度网络在随机初始化各层权重时，多次试验的初始化权重是一样的，结果可以复现
torch.manual_seed(args.seed)
# torch.cuda.manual_seed()为当前GPU设置随机种子
torch.cuda.manual_seed(args.seed)

# 如果训练
# 为模式"train" and "testall"创建记录器
if args.mode == "train":
    # 判断某一路径是否为目录
    if not os.path.isdir(args.logdir):
        # 创建目录
        os.mkdir(args.logdir)
    # strftime()(格式化日期时间的函数)
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    # 构建SummaryWriter(使用tensorboardx进行可视化)
    print("creating new summary file")
    # logger = SummaryWriter(args.logdir)
    logger = (args.logdir)

# sys.argv[]是用来获取命令行参数的，sys.argv[0]表示代码本身文件路径
# sys.argv[1:]表示从第二个参数到最后结尾
print("argv:", sys.argv[1:])
print_args(args)

# 构建MVSDataset和DatasetLoader
# 训练时调用dtu_yao.py,测试时为dtu_yao_eval.py
MVSDataset = find_dataset_def(args.dataset)  # 获取MVSDataset对象

train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
# drop_last (bool, optional) – 当样本数不能被batchsize整除时，最后一批数据是否舍弃（default: False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# 载入MVSNet, mvsnet_loss，优化器
model = MVSNet(refine=False)  # refine: 是否要深度图边缘优化
# 多个GPU来加速训练
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()
# 载入损失函数
model_loss = mvsnet_loss
# 载入优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


start_epoch = 0
# 如果之前有训练模型，从上次末尾或指定的模型继续训练，载入参数
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# 训练函数
def train():
    # 设置milestone动态调整学习率
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    # MultiStepLR是一个非常常见的学习率调整策略，它会在每个milestone时，将此前学习率乘以gamma
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)
    # 对于每个epoch训练，args.epochs决定训练周期
    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        # scheduler.step()是对lr进行调整
        # 在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # 对于每个batch数据进行训练
        for batch_idx, sample in enumerate(TrainImgLoader):
            # 计时
            start_time = time.time()
            # 计算当前总体step
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            # 隔summary_freq做记录，summary_freq为正
            # %	取模 - 返回除法的余数
            # == 比较对象是否相等，相等返回ture
            do_summary = global_step % args.summary_freq == 0
            # train_sample(),输出训练中的信息(loss和图像信息)
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            # 记录损失
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time))

        # 每个epoch后训练完保存模型
        # torch.save(state, dir)
        # state可以用字典，保存参数
        # 其中dir表示保存文件的绝对路径+保存文件名，如'/home/q/Desktop/modelpara.pth'
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # 每轮模型训练完进行验证
        # 主要存储loss那些信息，方便计算均值输出到fulltest
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    # train模式
    model.train()
    # 优化器梯度清零开始新一次的训练
    optimizer.zero_grad()
    # 将所有Tensor类型的变量使用cuda计算
    sample_cuda = tocuda(sample)
    # 真实的深度图
    depth_gt = sample_cuda["depth"]
    # mask用于将没有深度的地方筛除掉，不计算loss
    mask = sample_cuda["mask"]
    # 输入模型计算深度图
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    # MVSNet得到的深度估计图
    depth_est = outputs["depth"]
    # 计算估计深度和真实深度之间的损失，mask用于选取有深度值的位置，只用这些位置的深度真值计算loss
    loss = model_loss(depth_est, depth_gt, mask)
    # 反向传播，计算当前梯度；
    loss.backward()
    # 根据梯度更新网络参数
    optimizer.step()
    # 保存训练得到的loss
    scalar_outputs = {"loss": loss}
    # depth_est * mask：深度图估计(滤除掉本来就没有深度的位置)
    # depth_gt：深度图真值
    # ref_img：要估计深度的ref图
    # mask：mask图，0-1二值图，为1代表这里有深度值
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        # 预测图和真值图的绝对差值
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        # 绝对平均深度误差：mean[abs(est - gt)]
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        # 整个场景深度估计误差大于2mm的值，mean[abs(est - gt) > threshold]
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss = model_loss(depth_est, depth_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
