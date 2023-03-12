import os
import shutil

from datasets.data_io import read_pfm

if __name__ == '__main__':
    # 修改image图片的名称
    # directory = "/home/ubuntu/qhl/mvsNet_pytorch/data/self_made/scan3/cams"
    # for i in range(116):
    #     j = 8361 + i
    #     shutil.copyfile(os.path.join(directory, 'IMG_' + str(j) + '.txt'), os.path.join(directory, '%08d_cam.txt' % i))

    # 读取pfm深度图文件
    # a = read_pfm('/home/ubuntu/qhl/mvsNet_pytorch/self_made_out/scan2/depth_est/00000000.pfm')
    # print(a)

    # 将cam文件中的内参除以4
    for i in range(116):
        lists = []
        with open("/home/ubuntu/qhl/mvsNet_pytorch/data/scan4/cams/%08d_cam.txt" % i, 'r') as f:
            lines = f.readlines()
            for j in range(len(lines)):
                if j == 7 or j == 8:
                    line = ' '.join([str(float(tmp)/4) for tmp in lines[j].split()])
                    lists.append(line + '\n')
                else:
                    lists.append(lines[j])
        with open("/home/ubuntu/qhl/mvsNet_pytorch/data/scan4/cams/%08d_cam.txt" % i, "w") as f:
            for line in lists:
                f.write(line)
