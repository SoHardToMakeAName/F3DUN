import scipy.io as sio
import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
import random

def crop_image(image, size, samples, factor):
    h, w = image.shape[0], image.shape[1]
    pairs = []
    for i in range(samples):
        x = random.randint(0, h-size-1)
        y = random.randint(0, w-size-1)
        hr_crop = image[x:x+size, y:y+size]
        lr_crop = cv2.resize(hr_crop, dsize=(size//factor, size//factor), interpolation=cv2.INTER_CUBIC)
        pairs.append((lr_crop, hr_crop))
    return pairs

def crop_image2(msi, rgb, size, stride, factor):
    if msi.shape[0] != rgb.shape[0] or msi.shape[1] != rgb.shape[1]:
        raise Exception("Size mismatch: msi: {}, rgb: {}".format(msi.shape[:2], rgb.shape[:2]))
    h, w = msi.shape[0], msi.shape[1]
    pairs = []
    for x in range(0, h-size, stride):
        for y in range(0, w-size, stride):
            hr_msi = msi[x:x+size, y:y+size]
            lr_msi = cv2.resize(hr_msi, dsize=(size//factor, size//factor), interpolation=cv2.INTER_CUBIC)
            hr_rgb = rgb[x:x+size, y:y+size]
            lr_rgb = cv2.resize(hr_rgb, dsize=(size//factor, size//factor), interpolation=cv2.INTER_CUBIC)
            pairs.append((lr_msi, hr_msi, lr_rgb, hr_rgb))
    return pairs

# root = r"/data_c/lzq0330/dataset/CAVE"
root = r"F:\dataset\CAVE"
scale_factor = 4
filedirs = sorted(os.listdir(root))
test_dirs = filedirs[:12]
train_dirs = filedirs[12:]
print(test_dirs, len(test_dirs), train_dirs, len(train_dirs))
test_tar = r"F:\dataset\CaveL_x{}".format(scale_factor)
train_tar = r"F:\dataset/CaveL_x{}/train/train".format(scale_factor)
val_tar = r"F:\dataset\CaveL_x{}/eval/eval".format(scale_factor)
os.makedirs(train_tar, exist_ok=True)
os.makedirs(val_tar, exist_ok=True)
random_patches = 24

# prepare test dataset
ms_lst, gt_lst, rgb_lst, rgb_gt_lst = [], [], [], []
for dir in test_dirs:
    bands = sorted(glob.glob(os.path.join(root, dir, dir)+'/*.png'))
    rgb_file = glob.glob(os.path.join(root, dir, dir)+'/*.bmp')[0]
    rgb = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb / 255.0
    msi_data = []
    for band in bands:
        band_data = cv2.imread(band, cv2.IMREAD_ANYDEPTH)
        band_data = band_data/65535.0
        msi_data.append(band_data)
    msi = np.array(msi_data, dtype=np.float32)
    msi = msi.transpose((1, 2, 0))
    gt_lst.append(msi)
    msi_resized = cv2.resize(msi, dsize=(msi.shape[0]//scale_factor, msi.shape[1]//scale_factor), interpolation=cv2.INTER_CUBIC)
    ms_lst.append(msi_resized)
    rgb_gt_lst.append(rgb)
    rgb_lst.append(cv2.resize(rgb, dsize=(rgb.shape[0]//scale_factor, rgb.shape[1]//scale_factor), interpolation=cv2.INTER_CUBIC))
data = {'ms': ms_lst, 'gt': gt_lst, 'rgb': rgb_lst, 'rgb_gt': rgb_gt_lst}
sio.savemat(os.path.join(test_tar, "CaveL_test_x{}.mat".format(scale_factor)), data, format='5', do_compression=False)

# prepare training dataset
count = 0
for dir in train_dirs:
    bands = sorted(glob.glob(os.path.join(root, dir, dir)+'/*.png'))
    if len(bands) == 0:
        continue
    rgb_file = glob.glob(os.path.join(root, dir, dir)+'/*.bmp')[0]
    rgb = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb / 255.0
    msi_data = []
    for band in bands:
        band_data = cv2.imread(band, cv2.IMREAD_ANYDEPTH)
        band_data = band_data/65535.0
        msi_data.append(band_data)
    msi = np.array(msi_data, dtype=np.float32)
    msi = msi.transpose((1, 2, 0))
    count = 0
    if scale_factor == 8:
        size, stride = 192, 36
    elif scale_factor == 4:
        size, stride = 96, 72
    else:
        size, stride = 64, 32
    patches = crop_image2(msi, rgb, size, stride, scale_factor)
    for i in range(len(patches)):
        data = {'ms': patches[i][0], 'gt': patches[i][1], 'rgb': patches[i][2], 'rgb_gt': patches[i][3]}
        if random.random() < 0.1:
            sio.savemat(os.path.join(val_tar, "{}_patch_{}.mat".format(dir, count)), data, format='5',
                            do_compression=False)
        else:
            sio.savemat(os.path.join(train_tar, "{}_patch_{}.mat".format(dir, count)), data, format='5',
                            do_compression=False)
        count += 1
    print("generate {} patches!".format(count))


# mat = sio.loadmat(r"D:\Code\Python-code\dataset\CAVE_x4\train\watercolors_ms_patch_72.mat")
# rgb = mat['rgb_gt']
# plt.imshow(rgb)
# plt.show()
# for i in range(31):
#     plt.imshow(mat['gt'][:, :, i], cmap='gray')
#     plt.show()