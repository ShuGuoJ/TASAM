'''训练模型文件'''
import torch
from torch import nn, optim
from scipy.io import loadmat
import argparse
from Trainer import SpectralTrainer as Trainer
import numpy as np
from visdom import Visdom
import os
from HSIDataset import HSIDataset
from Model.module import SpectralNet as Net
from tqdm import tqdm
from Monitor import GradMonitor
from utils import summary
from torch.utils.data import DataLoader
import sys
import time


def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(file)
        list_name.append(file_path)
    return list_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help='Dataset name')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU ID')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='EPOCH')
    parser.add_argument('--ratio', type=float, default=0.01,
                        help='RATIO OF SAMPLE PER CLASS ')
    parser.add_argument('--run', type=int, default=10,
                        help='RUN')
    parser.add_argument('--patch', type=int, default=7,
                        help='PATCH SIZE')
    parser.add_argument('--batch', type=int, default=64,
                        help='BATCH SIZE')
    parser.add_argument('--hsz', type=int, default=64,
                        help='HIDDEN SIZE')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='NUM WORKER')
    parser.add_argument('--seed', type=int, default=666,
                        help='RANDOM SEED')
    parser.add_argument('--n_layer', type=int, default=3,
                        help='LAYER NUMBER')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--skip', action='store_true',
                        help='A skip connection between features before and after Gabor operation')
    arg = parser.parse_args()

    def train(name):
        basename = name.split('.')[0]
        # 可视化训练过程的损失和精度曲线
        # viz = Visdom(port=17000, env=basename)
        print(name)
        # save_root = 'models/{name}/'.format(name=str(i))
        save_root = f'models/{basename}/' + ('skip_' if arg.skip else '') + \
                    f'{arg.hsz}_units_{arg.lr:.1E}_lr_{arg.epoch}_epoches_{time.strftime("%Y_%m_%d_%H_%M", time.localtime())}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        data_path = data_dictory_path + name

        # 读取原始图像和gabor数据
        assert os.path.exists(data_path)
        m = loadmat(data_path)
        data = m['data']
        data_gabor = m['data_gabor']
        data_gabor = data_gabor.astype(float)
        data = data.astype(float)
        bands = data.shape[-1]
        monitor = GradMonitor()

        for r in range(arg.run):
            # 读取训练和测试真实标记
            gt_path = 'trainTestSplit/{}/ratio{}_run{}.mat'.format(basename, arg.ratio, r)
            assert os.path.exists(gt_path)
            m = loadmat(gt_path)
            tr_gt, te_gt = m['train_gt'], m['test_gt']
            # 构造训练和测试数据集
            tr_dataset = HSIDataset(data, data_gabor, tr_gt, arg.patch, is_normalize=True)
            print(len(tr_dataset))
            te_dataset = HSIDataset(data, data_gabor, te_gt, arg.patch, is_normalize=True)
            print(len(te_dataset))
            tr_loader = DataLoader(tr_dataset, batch_size=arg.batch, num_workers=arg.num_workers, shuffle=True,
                                   drop_last=True, pin_memory=False)
            te_loader = DataLoader(te_dataset, batch_size=256*10, num_workers=arg.num_workers, shuffle=False,
                                   pin_memory=False)

            # 模型，损失函数和优化器设置
            net = Net(arg.patch ** 2, arg.hsz, te_gt.max(), bands, arg.n_layer, skip=arg.skip)
            trainer = Trainer(net)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=arg.lr)
            schduler = optim.lr_scheduler.MultiStepLR(optimizer, [800, 1300], gamma=0.5)
            if r == 0:
                print(summary(net))

            # 可视化训练过程的损失和精度曲线
            # viz.line([[0., 0., 0.]], [0],
            #          win=f'{r}th loss with {arg.hsz} units, {arg.lr:.1E} lr, {arg.epoch} epoches',
            #          opts={
            #              'title': f'{r}th loss with {arg.hsz} units, {arg.lr:.1E} lr, {arg.epoch} epoches',
            #              'legend': ['tr_loss', 'te_loss', 'acc']})
            # viz.line([0.], [0],
            #          win=f'{r}th grad with {arg.hsz} units, {arg.lr:.1E} lr, {arg.epoch} epoches',
            #          opts={
            #              'title': f'{r}th grad with {arg.hsz} units, {arg.lr:.1E} lr, {arg.epoch} epoches',
            #              'legend': ['grad']})

            pbar = tqdm(range(arg.epoch))
            max_acc = 0
            te_loss, acc = 0, 0
            metrics = [[0] * arg.epoch, [0] * arg.epoch, [0] * arg.epoch]
            for epoch in pbar:
                pbar.set_description_str('Epoch: {}'.format(epoch))
                tr_loss = trainer.train(tr_loader, optimizer, criterion, device, monitor.clear())
                schduler.step()
                if epoch % 10 == 0:
                    te_loss, acc = trainer.evaluate(te_loader, criterion, device)
                pbar.set_postfix_str('train loss: {}, test loss: {}, acc: {}'.format(tr_loss, te_loss, max_acc))

                # 可视化训练过程的损失和精度曲线
                # viz.line([[tr_loss, te_loss, acc]], [epoch], win=f'{r}th loss with {arg.hsz} units, {arg.lr:.1E} lr, {arg.epoch} epoches', update='append')
                # viz.line([*monitor.get()], [epoch], win=f'{r}th grad with {arg.hsz} units, {arg.lr:.1E} lr, {arg.epoch} epoches', update='append')

                if acc > max_acc:
                    max_acc = acc
                    trainer.save(os.path.join(save_root, 'best_{}_{}.pkl'.format(arg.ratio, r)))
                metrics[0][epoch], metrics[1][epoch], metrics[2][epoch] = tr_loss, te_loss, acc

            np.savetxt(f'{save_root}/metrics_{arg.ratio}_{r}.txt', np.array(metrics, dtype=float), fmt='%.6f',
                       delimiter=',', newline='\n')

            del tr_loader
            del te_loader
            del tr_dataset
            del te_dataset


    device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
    print(device)
    # viz = Visdom(port=17000)
    # 设置随机种子
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    torch.backends.cudnn.deterministic = True
    data_dictory_path = "../tree-data_gabor/"

    if arg.name is None:
        allpath = []
        allpath = listdir(data_dictory_path, allpath)
        for i in allpath:
            train(i)
    else:
        train(arg.name)

    print('-'*5 + 'FINISH' + '-'*5)
