from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode == "test"
        self.metas = self.build_list()

    #  TODO 构建视点匹配列表，最终meta长度为1078，与train相比，少了光照变化的数据
    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    # TODO 读取相机参数：内参除4，最终生成的深度图也下采样4倍
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # 内参除4
        intrinsics[:2, :] /= 4
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval
    # 读取图片数据
    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        # assert np_img.shape[:2] == (1200, 1600)
        # assert np_img.shape[:2] == (756, 1008)
        # crop to (1184, 1600),裁掉下方的16个像素，图像尺寸变为1184*1600，裁剪后不需要修改内参
        np_img = np_img[:-20, :-16]  # do not need to modify intrinsics if cropping the bottom part
        return np_img
    # 读取img的深度图信息
    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        # (5, 3, 1184, 1600) 测试的时候有5张图像，读的时候每张被裁剪掉了下面16像素
        imgs = []
        mask = None
        depth = None
        # 深度假设范围，仍然是从425开始每隔2.5取一个数，一共192个
        depth_values = None
        # 5个投影矩阵，注意内参除了4倍
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # {:0>8}数字补零 (填充左边, 宽度为8)
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            # 将图片读取进来
            imgs.append(self.read_img(img_filename))
            # 读取相机参数文件，包括：内参、外参、深度最小值、深度间隔
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_testing/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
