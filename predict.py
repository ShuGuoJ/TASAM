'''
测试文件
'''
from scipy.io import loadmat, savemat
import numpy as np
import argparse
from Trainer import SpectralTrainer as Trainer
from HSIDataset import HSIDataset
import torch
import os
from Model.module import SpectralNet as Net
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import utils


def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):  
        file_path = os.path.join(file)  
        list_name.append(file_path)  
    return list_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help='Dataset name')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID')
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
    parser.add_argument('--time', type=str, required=True,
                        help='Checkpoint time')
    parser.add_argument('--epoch', type=int, default=1500,
                        help='Epoch')
    arg = parser.parse_args()

    def test(name):
        basename = name.split('.')[0]
        print(name)
        data_path = data_dictory_path + name
        acc_list = []

        # 读取数据
        assert os.path.exists(data_path)
        m = loadmat(data_path)
        data = m['data']
        data_gabor = m['data_gabor']
        data = data.astype(float)
        bands = data.shape[-1]
        gt = m['map']
        gt = gt.astype(int)
        h, w = gt.shape
        dataset = HSIDataset(data, data_gabor, np.ones_like(gt), arg.patch)
        data_loader = DataLoader(dataset, batch_size=arg.batch, num_workers=arg.num_workers, shuffle=False)

        for r in range(arg.run):
            save_root = 'prediction_otm/{name}/'.format(name=str(basename))
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            # 模型构造
            net = Net(arg.patch ** 2, arg.hsz, gt.max(), bands, arg.n_layer)
            net_path = 'models/{}/{}_units_{:.1E}_lr_{}_epoches_{}/best_{}_{}.pkl'.format(basename, arg.hsz, arg.lr,
                                                                                          arg.epoch, arg.time, arg.ratio, r)
            # 加载模型权重
            net.load_state_dict(torch.load(net_path))
            trainer = Trainer(net)

            preds, prob = trainer.predict(data_loader, device)
            preds = preds.reshape(h, w)
            prob = prob.reshape(h, w, -1)
            preds = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
            prob = prob if isinstance(prob, np.ndarray) else prob.cpu().numpy()

            gt_path = 'trainTestSplit/{}/ratio{}_run{}.mat'.format(basename, arg.ratio, r)
            m = loadmat(gt_path)
            te_gt = m['test_gt']
            indices = np.nonzero(te_gt != 0)
            acc = np.sum(te_gt[indices]==preds[indices]) / len(indices[0])
            print(acc)
            acc_list.append(acc)
            savemat(os.path.join(save_root, '{}.mat'.format(r)), {'pred': preds, 'acc': acc, 'prob': prob})

        print(f'mean acc: {np.mean(acc)}')

    device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
    # 设置随机种子
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    torch.backends.cudnn.deterministic = True
    data_dictory_path = '../tree-data_gabor/'

    if arg.name is None:
        allpath = []
        allpath = listdir(data_dictory_path, allpath)
        for i in allpath:
            test(i)
    else:
        test(arg.name)

    print('-' * 5 + 'FINISH' + '-' * 5)

