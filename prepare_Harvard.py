import scipy.io as sio
import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
import random

def random_crop(hsi_data, crop_size, crop_num, scale_factor):
    h, w = hsi_data.shape[0], hsi_data.shape[1]
    lst = []
    for i in range(crop_num):
        x = random.randint(0, h-crop_size-1)
        y = random.randint(0, w-crop_size-1)
        gt = hsi_data[x:x+crop_size, y:y+crop_size]
        ms = cv2.resize(gt, dsize=(crop_size//scale_factor, crop_size//scale_factor), interpolation=cv2.INTER_CUBIC)
        lms = cv2.resize(ms, dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        data = {'ms': ms, 'gt': gt, 'ms_bicubic': lms}
        lst.append(data)
    return lst

# def random_crop2(hsi_data, rgb_data, crop_size, stride, scale_factor):
#     h, w = rgb_data.shape[0], rgb_data.shape[1]
#     lst = []
#     for x in range(0, h-crop_size-1, stride):
#         for y in range(0, w-crop_size, stride):
#             rgb_gt = rgb_data[x:x+crop_size, y:y+crop_size]
#             rgb = cv2.resize(rgb_gt, dsize=(crop_size//scale_factor, crop_size//scale_factor), interpolation=cv2.INTER_CUBIC)
#             gt = hsi_data[x:x+crop_size, y:y+crop_size]
#             ms = cv2.resize(gt, dsize=(crop_size//scale_factor, crop_size//scale_factor), interpolation=cv2.INTER_CUBIC)
#             data = {'ms': ms, 'gt': gt, 'rgb': rgb, 'rgb_gt': rgb_gt}
#             lst.append(data)
#     print("generate {} patches".format(len(lst)))
#     return lst


# generate train set
hsi_dir = r"D:\Code\Python-code\dataset\Harvard\train"
hsi_files = os.listdir(hsi_dir)
scale = 8
tar_dir = r"D:\Code\Python-code\dataset\Harvard_x{}/train/train".format(scale)
val_dir = r"D:\Code\Python-code\dataset\Harvard_x{}/eval/eval".format(scale)
os.makedirs(tar_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
count = 0
for hsi_file in hsi_files:
    hsi_data = sio.loadmat(os.path.join(hsi_dir, hsi_file))
    hsi_name, _ = os.path.splitext(hsi_file)
    hsi_data = hsi_data['ref']
    hsi_data = hsi_data / np.max(hsi_data)
    if scale == 4:
        crop_size = 96
    else:
        crop_size = 128
    lst = random_crop(hsi_data, crop_size=crop_size, crop_num=72, scale_factor=scale)
    # lst = random_crop2(hsi_data, rgb_data, crop_size=crop_size, stride=stride, scale_factor=scale)
    for data in lst:
        count += 1
        if random.random() <= 0.1:
            sio.savemat(os.path.join(val_dir, "{}_patch_{}.mat".format(hsi_name, count)), data, format='5',
                            do_compression=False)
        else:
            sio.savemat(os.path.join(tar_dir, "{}_patch_{}.mat".format(hsi_name, count)), data, format='5',
                        do_compression=False)
    print("{} done".format(hsi_name))
print("{} patches generated!".format(count))

hsi_dir = r"D:\Code\Python-code\dataset\Harvard\test"
hsi_files = os.listdir(hsi_dir)
lst_ms, lst_gt, lst_lms = [], [], []
test_size = 512
for hsi_file in hsi_files:
    hsi_data = sio.loadmat(os.path.join(hsi_dir, hsi_file))
    hsi_name, _ = os.path.splitext(hsi_file)
    hsi_data = hsi_data['ref']
    hsi_data = hsi_data / np.max(hsi_data)
    h, w, _ = hsi_data.shape
    for x in range(0, h-test_size, test_size):
        for y in range(0, w-test_size, test_size):
            gt = hsi_data[x:x+test_size, y:y+test_size, :]
            ms = cv2.resize(gt, dsize=(test_size//scale, test_size//scale), interpolation=cv2.INTER_CUBIC)
            lms = cv2.resize(ms, dsize=(test_size, test_size), interpolation=cv2.INTER_CUBIC)
            lst_ms.append(ms); lst_gt.append(gt); lst_lms.append(lms)
            print(ms.shape, gt.shape, lms.shape)
data_dict = {'ms': lst_ms, 'gt': lst_gt, 'ms_bicubic': lst_lms}
sio.savemat(r"D:\Code\Python-code\dataset\Harvard_x{0}\Harvard_test_x{0}.mat".format(scale), data_dict, format='5', do_compression=False)