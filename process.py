import os
import shutil

if __name__ == '__main__':
    directory = "/home/ubuntu/qhl/mvsNet_pytorch/data/self_made/scan3/cams"
    for i in range(116):
        j = 8361 + i
        shutil.copyfile(os.path.join(directory, 'IMG_' + str(j) + '.txt'), os.path.join(directory, '%08d_cam.txt' % i))

