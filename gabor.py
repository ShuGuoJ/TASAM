'''
提取高光谱树种数据的3D Gabor特征
'''
import torch
import copy
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import cv2 as cv
from torch.nn import grad  # noqa: F401
import os
from scipy.io import loadmat,savemat
import numpy as np
from torch.nn.modules.container import ModuleList
from sklearn.decomposition import PCA
from argparse import ArgumentParser


# 定3D_Gabor方法
def get_3d_gabor_filter(f, theta, phi, ratio, x, y, d):
    sigma = ratio / f

    u = f * np.sin(theta) * np.cos(phi)
    v = f * np.sin(theta) * np.sin(phi)
    w = f * np.cos(theta)

    [X, Y, Z] = np.meshgrid(np.arange(-x // 2 + 1, x // 2 + 1),
                            np.arange(-y // 2 + 1, y // 2 + 1),
                            np.arange(-d // 2 + 1, d // 2 + 1))

    prefix = 1 / (((2 * np.pi) ** 1.5) * sigma ** 3)
    gaussian = prefix * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sigma ** 2))
    # modulate = np.exp(1j*2*np.pi*(u*X + v*Y + w*Z))
    cosine = np.cos(2 * np.pi * (u * X + v * Y + w * Z))

    # g = gaussian * modulate
    g_real = gaussian * cosine
    # io.savemat("save.mat", {"result1": g_real})
    g_real = np.swapaxes(np.swapaxes(g_real, 0, 2), 1, 2)

    return g_real


def get_3d_gabor_filter_bank(nScale=1, M=13, x=3, y=3, d=13):
    f = 1 / (2 * nScale)

    theta = np.array([0, 1 / 4 * np.pi, 1 / 2 * np.pi, 3 / 4 * np.pi])
    phi = np.array([0, 1 / 4 * np.pi, 1 / 2 * np.pi, 3 / 4 * np.pi])

    g_filter_bank = np.zeros([M, d, x, y])
    counter = 0
    for i in [1, 3]:
        for j in range(4):
            g_filter_bank[counter] = get_3d_gabor_filter(f=f, theta=theta[i], phi=phi[j], ratio=1, x=x, y=y, d=d)
            counter += 1
    g_filter_bank = torch.from_numpy(np.float32(g_filter_bank))
    return g_filter_bank


def Gabor_3D(x:Tensor):
    # 生成gabor核 torch.Size([13, 1, 16, 5, 5])
    g = get_3d_gabor_filter_bank(nScale=1, M=13, x=5, y=5, d=10)
    # print(g.size())
    g = g.unsqueeze(1)
    print(x.shape)
    x = (Tensor.float(x)).unsqueeze(0).unsqueeze(0)
    print(x.shape)
    x = nn.functional.conv3d(input=x, weight=g, padding=[0, 2, 2])
    print(x.shape)
    x = x.squeeze(0).squeeze(1)
    print(x.shape)
    return x


def Gabor_2D(x:Tensor):
    x = x.numpy()
    b = x.shape[0]
    for index_i in range(b):
        # paojie
        # retval = cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]])
        # Ksize 是一个元组
        retval = cv.getGaborKernel(ksize=(111, 111), sigma=10, theta=b, lambd=10, gamma=1.2)
        image1 = x[:, :, index_i]
        # dst   =   cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
        result = cv.filter2D(image1, -1, retval)
        x[:, :, index_i] = result
    x = torch.from_numpy(x)
    return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def entropy(hsi): #
    '''
    计算各波段的信息熵
    传入参数:hsi高光谱数据
    返回参数：ent各波段信息熵结果
    '''

    # 获取数据维度信息
    M, N, b = hsi.shape[0:3]
    # 定义ent变量存储信息熵结果
    ent = np.zeros(b)
    # 逐一波段求信息熵
    for i in range(b):
        # 获取当前波段数据
        band = hsi[:, :, i]
        band = band.reshape(N*M).tolist()
        # 定义频数、频率对应变量
        countDict = dict()
        proportitionDict = dict()
        # 利用集合去重
        band_set = set()
        for l in band:
            # print(l)
            band_set.add(l)
        # 计算信息熵
        for k in band_set:
            # 对集合每个非重复元素求得其在该波段的频数、频率
            countDict[k] = band.count(k)
            proportitionDict[k] = band.count(k)/len(band)
            # 根据信息熵公式求信息熵
            logp = np.log2(proportitionDict[k])
            ent[i] -= proportitionDict[k] * logp
    # 输出结果，并返回
    print(ent)
    print(len(ent))
    return ent


def selectBand(hsi, ent, threshold):
    '''
    波段选择：根据设置阈值，选择信息熵大于等于阈值的波段
    传入参数：hsi 高光谱数据,
            ent 各波段信息熵,
            threshold：设定阈值
    返回参数：selectband 选取的波段
            selecthsi 波段选择后的数据
    '''
    M, N, b = hsi.shape[0:3]
    selectband = list(range(b))
    for i in range(len(ent)):
        if ent[i] < threshold:
            selectband.remove(i)

    selecthsi = hsi[:, :, selectband]
    print('选取的波段为：', selectband)
    return selectband, selecthsi


def selectBandByVar(hsi, var, select_num):
    '''
    波段选择：根据设置阈值，选择信息熵大于等于阈值的波段
    传入参数：hsi 高光谱数据,
            var 各波段方差,
            select_num：设定选择波段数
    返回参数：selectband 选取的波段
            selecthsi 波段选择后的数据
    '''
    M, N, b = hsi.shape[0:3]
    selectband = []
    var = var.tolist()
    print('var:', var)
    sorted_var = sorted(var, reverse = True)
    print('sorted_var:', sorted_var)
    for i in range(select_num):
        selectband.append(var.index(sorted_var[i]))

    selecthsi = hsi[:, :, selectband]
    print('选取的波段为：', selectband)
    print('选取的波段方差为：', sorted_var)
    return selectband, selecthsi


def pcadata(x, band_num):
    h,w = x.shape[0:2]
    # PCA 降维
    pca = PCA(60, whiten=True)
    x_pca = pca.fit_transform(x.reshape(h * w, -1))
    x_pca = x_pca.reshape(h, w, -1)
    return x_pca


def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(file)
        list_name.append(file_path)
    return list_name


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='DATA PATH', default='')
    parser.add_argument('-o', '--output', type=str, help='SAVING PATH', default='../data_gabor')
    args = parser.parse_args()
    return args


def main(cfg):
    save_root = cfg.output
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print('-' * 5 + 'START' + '-' * 5)
    dataname = os.path.basename(cfg.input).split('.')[0]
    print(dataname)
    m = loadmat(cfg.input)
    data = m['data']
    map = m['map']
    print(data.shape)
    data = data.transpose(2, 1, 0)
    print(data.shape)
    x = torch.from_numpy(data)
    print(x.shape)
    data_gabor = Gabor_3D(x)
    print(data_gabor.shape)
    for i in range(8):
        d = data_gabor[i, :, :, :]
        d = d.squeeze(0)
        d = d.numpy()
        d = d.transpose(2, 1, 0)
        if i == 0:
            newdata = d
        else:
            newdata = np.concatenate((newdata, d), axis=2)
    data = data.transpose(2, 1, 0)

    data_gabor = newdata
    print("data_gabor.shape:", newdata.shape)
    select_num = 60  # 选取波段数量
    pca_hsi = pcadata(data_gabor, select_num)
    print("save_data.shape:", data.shape)
    print("save_data_gabor_hsi.shape:", pca_hsi.shape)

    savemat('{savedir}/{name}_pca_after_gabor.mat'.format(savedir=save_root, name=str(dataname)),
            {'data': data, 'data_gabor': pca_hsi, 'map': map})
    print('{name}_pca_after_gabor is saved !'.format(name=str(dataname)))
    print('-' * 5 + 'FINISH ' + '-' * 5)


if __name__ == '__main__':
    args = get_args()
    main(cfg=args)


