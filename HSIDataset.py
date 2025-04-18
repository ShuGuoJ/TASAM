'''
高光谱树种数据集文件
'''
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.decomposition import PCA


class HSIDataset(Dataset):
    def __init__(self, x, x_gabor, gt, patch, is_normalize=True):
        super(HSIDataset, self).__init__()
        # 正则化
        h, w = x.shape[:2]
        if is_normalize:
            scaler = StandardScaler()
            x_normalization = scaler.fit_transform(x.reshape((h * w, -1)))
            x = x_normalization.reshape((h, w, -1))
        x_mirror = self.addMirror(x, patch)
        x1_mirror = self.addMirror(x, patch=14)
        x2_mirror = self.addMirror(x, patch=28)
        if is_normalize:
            scaler = StandardScaler()
            x_normalization = scaler.fit_transform(x_gabor.reshape((h * w, -1)))
            x_gabor = x_normalization.reshape((h, w, -1))
        x_gabor_mirror = self.addMirror(x_gabor, patch)
        # PCA 降维
        # pca = PCA(3, whiten=True)
        # x_pca = pca.fit_transform(x.reshape(h * w, -1))
        # x_pca = x_pca.reshape(h, w, -1)
        # x_pca_mirror = self.addMirror(x_pca, patch)
        # self.x_pca = x_pca_mirror
        self.x = x_mirror
        self.x1 = x1_mirror
        self.x2 = x2_mirror
        self.x_gabor = x_gabor_mirror
        self.gt = gt
        self.patch = patch
        self.indices = tuple(zip(*np.nonzero(gt)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        l, c = self.indices[index]
        data = self.x[l: l + self.patch, c: c + self.patch]
        data1 = self.x1[l: l + 14, c: c + 14]
        data2 = self.x2[l: l + 28, c: c + 28]
        data_gabor = self.x_gabor[l: l + self.patch, c: c + self.patch]
        # data_pca = self.x_pca[l: l + self.patch, c: c + self.patch]
        y = self.gt[l, c]
        return torch.tensor(data, dtype=torch.float),torch.tensor(data1, dtype=torch.float),torch.tensor(data2, dtype=torch.float), torch.tensor(data_gabor, dtype=torch.float), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def addMirror(x, patch):
        patch_half = patch // 2
        x_mirror = np.zeros((x.shape[0] + 2 * patch_half, x.shape[1] + 2 * patch_half, x.shape[2]))
        x_mirror[patch_half:-patch_half, patch_half:-patch_half, :] = x
        for i in range(patch_half):
            # 填充左上部分
            x_mirror[:, i, :] = x_mirror[:, 2 * patch_half - 1, :]
            x_mirror[i, :, :] = x_mirror[2 * patch_half - 1, :, :]

            # 填充右下部分
            x_mirror[:, -i - 1, :] = x_mirror[:, i - 1 - 2 * patch_half, :]
            x_mirror[-i - 1, :, :] = x_mirror[i - 1 - 2 * patch_half, :, :]

        return x_mirror



# from scipy.io import loadmat
# m = loadmat('data/medicine.mat')
# data = m['data']
# gt = m['map']
# dataset = HSIDataset(data, gt, 5)
# x, x_pca, y = dataset[0]
# print(x.shape)
# print(x_pca.shape)
# print(y.shape)


