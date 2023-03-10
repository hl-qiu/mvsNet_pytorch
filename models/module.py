import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积+BN+ReLU
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


# 卷积+BN
class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


# 3d卷积+BN+ReLU
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


# 3d卷积+BN
class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


# TODO 未使用
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


# TODO 未使用
class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


'''单应性变换：从参考视角到去找源视角的像素坐标对应点的计算过程，可以用单应性矩阵来描述。将src图根据ref和src的投影矩阵，投影到ref视角下,得到特征体。
#  变换过程：已知参考视角的内外参数矩阵，深度信息，可以计算参考视角像素坐标的世界坐标系值，即将其投影到世界坐标系。
#  再通过已知的源视角内外参数矩阵，可以投影到源视角的相机坐标系，归一化后得到源视角的像素坐标。
最终，将在源视角对应点的特征（通道维度的所有数据）存放在参考视角的像素坐标位置处完成变换。
'''


#  单应性变换
def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]=》(B, 32, 160, 128) src图像的特征 此时的C已经是32维了
    # src_proj: [B, 4, 4] src图像的投影矩阵
    # ref_proj: [B, 4, 4] 参考图像的投影矩阵
    # depth_values: [B, Ndepth] 深度假设范围数组
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    # 阻止梯度计算，降低计算量
    with torch.no_grad():
        # src * ref.T，得到变换矩阵
        # .matmul(input, other, out = None) ：input 和 other 两个张量进行矩阵相乘
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        # 取左上角三行三列得到旋转变换,[B,3,3]
        rot = proj[:, :3, :3]
        # 取最后一列的上面三行得到平移变换,[B,3,1]
        trans = proj[:, :3, 3:4]
        # 按照ref图像维度构建一张空的平面，之后要做的是根据投影矩阵把src中的像素映射到这张平面上，也就是提取特征的坐标
        # y: Size([128, 160])
        # y: tensor([[  0.,   0.,   0.,  ...,   0.,   0.,   0.],
        #         [  1.,   1.,   1.,  ...,   1.,   1.,   1.],
        #         [  2.,   2.,   2.,  ...,   2.,   2.,   2.],
        #         ...,
        #         [125., 125., 125.,  ..., 125., 125., 125.],
        #         [126., 126., 126.,  ..., 126., 126., 126.],
        #         [127., 127., 127.,  ..., 127., 127., 127.]], device='cuda:0')
        # x: Size([128, 160])
        # x: tensor([[  0.,   1.,   2.,  ..., 157., 158., 159.],
        #         [  0.,   1.,   2.,  ..., 157., 158., 159.],
        #         [  0.,   1.,   2.,  ..., 157., 158., 159.],
        #         ...,
        #         [  0.,   1.,   2.,  ..., 157., 158., 159.],
        #         [  0.,   1.,   2.,  ..., 157., 158., 159.],
        #         [  0.,   1.,   2.,  ..., 157., 158., 159.]], device='cuda:0')
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        # 保证开辟的新空间是连续的(数组存储顺序与按行展开的顺序一致，transpose等操作是跟原tensor共享内存的)
        y, x = y.contiguous(), x.contiguous()
        # 将维度变换为图像样子
        # x: ：Size([20480])
        # x: tensor([  0.,   1.,   2.,  ..., 157., 158., 159.], device='cuda:0')
        y, x = y.view(height * width), x.view(height * width)   # 拉成一维
        # .ones_like(x)：返回一个填充了标量值1的张量，其大小与x相同
        # .stack: [3, H*W]，即([3, 20480])
        # xyz: tensor([[  0.,   1.,   2.,  ..., 158., 159., ...,0., 1., 2.,  ..., 157., 158., 159],
        #        [  0.,   0.,   0.,  ..., 0.,   0.,  1., 1., 1., ..., 127., 127., 127.],
        #        [  1.,   1.,   1.,  ...,   1.,   1.,   1.]], device='cuda:0')
        xyz = torch.stack((x, y, torch.ones_like(x)))
        # unsqueeze：升维，第0维添加维度1，则[3, 20480]变为[1, 3, 20480]
        # repeat：将原矩阵按 各维度 复制，得到[batch,3,20480]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)   # 表示将xyz[1,3,20480]按0维度复制batch遍，1维度复制1遍，2维度复制1遍
        # [B, 3, H*W] 先将坐标乘以旋转矩阵
        rot_xyz = torch.matmul(rot, xyz)
        # [B, 3, Ndepth, H*W] 再引入Ndepths维度，并将深度假设值填入这个维度
        # rot_depth_xyz: Size([B, 3, 192, 20480])
        # depth_values.view(batch, 1, num_depth,1): Size([4, 1, 192, 1])
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, 1)
        # 旋转变换后的矩阵+平移矩阵 -> 投影变换后的坐标
        # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
        # xy分别除以z进行归一化
        # [B, 2, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
        # F.grid_sample中的grid要求值在[-1,1]
        # [B, Ndepth, H*W] x方向按照宽度进行归一化
        # proj_x_normalized: Size([4, 192, 20480])
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        # y方向按照高度进行归一化
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        # 再把归一化后的x和y拼起来
        # [B, Ndepth, H*W, 2]
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
        grid = proj_xy
    # 根据变化后的坐标在源特征图检索对应的值，即为变化后的值
    # warped_src_fea: Size([4, 3, 24576, 160])
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    # 将上一步编码到height维的深度信息分离出来
    # warped_src_fea:  Size([4, 3, 192, 128, 160])
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    # 最终得到的可以理解为src特征图按照不同假设的深度值投影到ref后构建的特征体
    # [B, C, Ndepth, H, W]
    return warped_src_fea


# 深度回归：根据之前假设的192个深度经过网络得到的不同概率，乘以深度假设，求得期望
# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    # 最后在深度假设维度做了加法，运算后深度假设这一维度就没了，期望即为最终估计的深度图
    depth = torch.sum(p * depth_values, 1)
    return depth


# 测试代码，忽略
if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("../data/mvs_training/dtu/", '../lists/dtu/train.txt', 'train', 3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    # 图片尺寸缩小四倍=》(1/4)H,(1/4)W
    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    # 按第一维拆开
    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
