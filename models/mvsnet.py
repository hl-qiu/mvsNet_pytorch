import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

# 特征提取网络
# 将3通道的rgb图像转换为32维的高维深度特征，同时图像进行了4倍下采样
# 输入：[3, H, W]
# 输出：[32, H/4, W/4]，即为(32, 160, 128)
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32
        # 输入通道数，输出通道数，卷积核大小，卷积步长，padding
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x

# 代价体正则化网络
# 先进行3D卷积降维，再反卷积升维，过程中把每一步卷积和反卷积对应的代价体都加起来，实现跳跃连接
# 输入：[B, C, D, H/4, W/4]
# 输出：[B, 1, D, H/4, W/4]，即(B, 1, 192, 160, 128)
class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


# 深度图边缘优化残差网络
# 输入: [B, 4, H/4, W/4]，4是即为img有3通道，depth有1通道
# 输出: [B, 1, H/4, W/4]，即(B, 1, 160, 128)
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine
        # 特征提取网络
        self.feature = FeatureNet()
        # 代价体正则化网络
        self.cost_regularization = CostRegNet()
        # 深度图边缘优化残差网络
        if self.refine:
            self.refine_network = RefineNet()

    # 输入：图像、内外参数、采样的深度值
    def forward(self, imgs, proj_matrices, depth_values):
        # .unbind()：对某一个维度进行长度为1的切片，并将所有切片结果返回，组成列表
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        # 确定图像列表和参数列表长度相等
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # TODO step 1. 特征提取
        # 输入：每张图片[B, 3, H, W]
        # 输出：特征图[B, 32, H/4, W/4]，即为(32, 160, 128)
        features = [self.feature(img) for img in imgs]
        # ref为参考图像，src为源图像
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # TODO step 2. 通过单应性变化构建代价体
        # 将ref的32维特征和ref投过来的特征图通过方差构建原始的代价体
        # ref_volume：Size([4, 32, 192, 128, 160])
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        # del删除的是变量，而不是数据
        del ref_volume
        # zip()：将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        for src_fea, src_proj in zip(src_features, src_projs):
            # 单应性变换 =》 [B, C, Ndepth, H, W] ==》 [4, 3, 192, 128, 160]
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                # pow()：返回x的y次方的值。
                # the memory of warped_volume has been modified
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume
        # aggregate multiple feature volumes by variance
        # 通过公式计算方差得到合并的代价体(在实现里通过公式简化计算)
        # 最终的cost volume维度是[B, 32, 192, H/4, W/4]
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # TODO step 3. 代价体正则化，
        # 首先通过代价体正则化网络进行进一步信息聚合，最终得到的维度是[B, 1, 192, H/4, W/4]
        cost_reg = self.cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        # 通过squeeze将维度为1的维度去除掉，得到[B, 192, H/4, W/4]
        cost_reg = cost_reg.squeeze(1)
        # 通过Softmax函数，将深度维度的信息压缩为0～1之间的分布，得到概率体
        prob_volume = F.softmax(cost_reg, dim=1)
        # 回归得到深度图
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence：用于进行光度一致性校验，最终得到跟深度图尺寸一样的置信度图：
            # 简单来说就是选取上面估计的最优深度附近的四个点，再次通过depth regression得到深度值索引，
            # 再通过gather函数从192个深度假设层中获取index对应位置的数据
            # F.pad：参数pad定义了六个参数，表示对输入矩阵的后三个维度进行扩充
            # prob_volume_sum4：Size([4, 192, 128, 160])
            # prob_volume：Size([4, 192, 128, 160])
            # depth_index：Size([4, 1，128, 160])
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            # photometric_confidence：Size([4, 128, 160])
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # step 4. 深度图改进
        # 将原图和得到的深度图合并输入至优化残差网络，输出优化后的深度图
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}

# 由于是有监督学习，loss就是估计的深度图和真实深度图的smoothl1
# 唯一要注意的是，数据集中的mask终于在这发挥作用了，我们只选取mask>0.5，也就是可视化中白色的部分计算loss，只有这部分的点深度是有效的
def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
