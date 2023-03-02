from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for training)
# datapath: 数据集路径
# listfile: 数据列表(用哪些scan训练和测试都是提前定好的)
# mode: train or test
# nviews: 多视点总数(实现中取3=1ref+2src)
# ndepths: 深度假设数(默认假设192种不同的深度)
# interval_scale: 深度间隔缩放因子(数据集文件中定义了深度采样间隔是2.5，再把这个值乘以缩放因子，最终每隔2.5*1.06取一个不同的深度假设)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
    # 构建训练样本条目，最终的meta数组中共用27097条数据，每个元素如下：
    # # scan   light_idx      ref_view          src_view
    # # 场景    光照(0~6)  中心视点(估计它的深度)    参考视点
    # ('scan2', 0, 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
            scans = f.readlines()
            # rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    # 相机外参、相机内参、最小深度(都为425)、深度假设间隔(都为2.5)
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    # 将图像归一化到0～1(神经网络训练常用技巧，激活函数的取值范围大都是0～1，便于高效计算)
    # 79个不同的scan
    # 7种不同的光照
    # 每个scan有49个不同的中心视点
    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    # getitem(): 取一组用来训练的数据
    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        # imgs: 1ref + 2src（都归一化到0-1） (3, 3, 512, 640) 3个3channel的512*640大小的图片
        imgs = []
        # ref深度图的mask(0-1二值图)，用来选取真值可靠的点 (128, 160)
        mask = None
        # ref的深度图 (128, 160)
        depth = None
        # ref将来要假设的所有深度值 (从425开始每隔2.5取一个数，一共取192个)
        # 2.5还要乘以深度间隔缩放因子
        depth_values = None
        # proj_metrices: 3个4*4投影矩阵
        # 这里是一个视点就有一个投影矩阵，因为MVSNet中所有的投影矩阵都是相对于一个基准视点的投影关系，所以如果想建立两个视点的关系，他们两个都有投影矩阵
        # 投影矩阵按理说应该是33的，这里在最后一行补了[0, 0, 0, 1]为了后续方便计算，所以这里投影矩阵维度是44

        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "depth_values": depth_values,
                "mask": mask}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 128)
    item = dataset[50]

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/val.txt', 'val', 3,
                         128)
    item = dataset[50]

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]

    # test homography here
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask"].shape)

    ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
    src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, 5)]
    ref_proj_mat = item["proj_matrices"][0]
    src_proj_mats = [item["proj_matrices"][i] for i in range(1, 5)]
    mask = item["mask"]
    depth = item["depth"]

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx)))
    D = depth.reshape([-1])
    print("X", "D", X.shape, D.shape)

    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mats[0], X)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)
    import cv2

    warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
    warped[mask[:, :] < 0.5] = 0

    cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
    cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    cv2.imwrite('../tmp2.png', src_imgs[0][:, :, ::-1] * 255)
