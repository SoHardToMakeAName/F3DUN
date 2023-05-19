import numpy as np
import torch.utils.data as data
import torch
import h5py
import utils


class HSDataset(data.Dataset):
    def __init__(self, image_dir, augment=True, use_3D=False):

        h5data = h5py.File(image_dir, 'r')

        self.gt = h5data['gt'][...]
        self.bicubic = h5data['bicubic'][...]
        self.lr = h5data['lr'][...]

        h5data.close()

        # print(self.gt.shape)

        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):

        aug_num = 0

        if self.augment:
            index = index // self.factor
            aug_num = int(index % self.factor)

        ms, lms, gt = utils.data_augmentation(self.lr[index], mode=aug_num), \
                      utils.data_augmentation(self.bicubic[index], mode=aug_num), \
                      utils.data_augmentation(self.gt[index], mode=aug_num)

        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return self.gt.shape[0] * self.factor
