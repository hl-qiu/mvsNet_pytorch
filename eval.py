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
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Predict depth, filter, and fuse. May be different from the original implementation')
parser.add_argument('--model', default='mvsnet', help='select model')
# 指定数据集预处理方法
parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./self_made_out', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# 读取相机内参和外参
# 测试时参考图像用了5个视点
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        # rstrip()：删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符。
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # 相机参数读取是内参intrinsics要除4
    # TODO: assume the feature is 1/4 of the original image size
    intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


# 运行MVS模型保存深度图和置信度图
def save_depth():
    # TODO 首先构建MVSDataset和Loader
    MVSDataset = find_dataset_def(args.dataset)
    # 读取测试数据：imgs、内外参矩阵、初始化depth_values
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)

    # 构建Loader
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    # TODO 构建model
    model = MVSNet(refine=False)
    # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # 将模型对象转变为多GPU并行运算的模型
    # model = nn.DataParallel(model)
    model = nn.DataParallel(model)
    # torch.cuda.set_device([0, 1])
    model.cuda()

    # TODO 加载模型参数
    print("loading model {}".format(args.loadckpt))
    # step1：加载torch.save()保存的模型参数文件
    state_dict = torch.load(args.loadckpt)
    # step2：将参数加载到模型中
    model.load_state_dict(state_dict['model'])
    # 作用：不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是使用训练好的值。避免一旦test时的batch_size过小，被BN层影响结果。
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            # 输入：1ref + 4src，每个视点的投影矩阵，深度假设list
            # 输出：深度图，置信度图
            # 深度图里的数据都是668.08545, 559.7229这类的物理距离
            # 置信度里的数据是0～1之间的小数
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)  # tensor转为numpy
            del sample_cuda

            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # 保存深度图和置信度图：将模型输出的两张图保存成pfm格式文件
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"],
                                                                   outputs["photometric_confidence"]):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))

                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # 保存深度图
                save_pfm(depth_filename, depth_est)
                # 保存置信度图
                save_pfm(confidence_filename, photometric_confidence)


# 将ref的点投影到src上，再投影回来
# 参数：ref的深度图和内外参，src的深度图和内外参
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    # mask: 通过几何检验的mask图
    # depth_reprojected: 重投影后的深度图
    # x2d_src： ref这些像素在src上的坐标
    # y2d_src： ref这些像素在src上的坐标
    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


# 通过光度一致性约束和几何一致性约束过滤深度图
def filter_depth(scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # 加载相机参数
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # 加载参考图像
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # 加载参考图像的估计深度图
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # 加载参考图像的置信度图
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        # 置信度掩码: 置信度图 > 0.8
        photo_mask = confidence > 0.8

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # 计算几何掩码
        geo_mask_sum = 0
        for src_view in src_views:
            # 加载原图像的相机参数
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # 加载源图像的估计的深度图
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                        ref_extrinsics,
                                                                                        src_depth_est,
                                                                                        src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # 几何掩码: 至少3个src满足上面的几何一致性校验(重投影后像素偏移 < 1 & & 深度差 < 1 %)
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))
        # filter每张ref的x y depth，并赋予颜色
        # 最终融合生成最后的点云
        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':

    # TODO step1. 在outputs目录中保存所有的深度图和掩码
    save_depth()

    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    for scan in scans:
        scan_id = int(scan[4:])
        scan_folder = os.path.join(args.testpath, scan)
        out_folder = os.path.join(args.outdir, scan)
        print("ok")
        # TODO step2. 过滤保存的深度图与光度置信度图和进行几何约束
        filter_depth(scan_folder, out_folder, os.path.join(args.outdir, 'mvsnet{:0>3}_l3.ply'.format(scan_id)))
