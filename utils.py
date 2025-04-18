from torch import nn
import xlwt
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score
import os

def save_result(value,name):
    f = open("prediction/"+name+"/result.txt", 'a')  # 将要输出保存的文件地址
    f.write(str(value))
    f.write("\n")  # 换行

#     """
#     Save an array to a text file.
#
#
def summary(net: nn.Module):
    single_dotted_line = '-' * 40
    double_dotted_line = '=' * 40
    dash_line = '.' * 40
    star_line = '*' * 40
    content = []
    def backward(m: nn.Module, chain: list):
        children = m.children()
        params = 0
        chain.append(m._get_name())
        try:
            child = next(children)
            params += backward(child, chain)
            for child in children:
                params += backward(child, chain)
            # print('*' * 40)
            # print('{:>25}{:>15,}'.format('->'.join(chain), params))
            # print('*' * 40)
            parameters = m.named_parameters(recurse=False)
            try:
                name, p = next(parameters)
                content.append(dash_line)
                while True:
                    content.append('{:>25}{:>15}'.format(name, p.numel()))
                    params += p.numel()
                    name, p = next(parameters)
            except:
                # content.append(dash_line)
                pass
            if content[-1] is not star_line:
                content.append(star_line)
            content.append('{:>25}{:>15,}'.format('->'.join(chain), params))
            content.append(star_line)
        except:
            for p in m.parameters():
                if p.requires_grad:
                    params += p.numel()
            # print('{:>25}{:>15,}'.format(chain[-1], params))
            content.append('{:>25}{:>15,}'.format(chain[-1], params))
            pass
        finally:
            # for p in m.parameters():
            #     if p.requires_grad:
            #         params += p.numel()
            # # print('{:>25}{:>15,}'.format(chain[-1], params))
            # content.append('{:>25}{:>15,}'.format(chain[-1], params))
            pass
        chain.pop()
        return params
    # print('-' * 40)
    # print('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    # print('=' * 40)
    content.append(single_dotted_line)
    content.append('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    content.append(double_dotted_line)
    params = backward(net, [])
    # print('=' * 40)
    # print('-' * 40)
    content.pop()
    content.append(single_dotted_line)
    print('\n'.join(content))
    return params


def save_metrics_as_xls(gt, prediction, save_root):
    book = xlwt.Workbook()
    # 创建sheet
    sheet = book.add_sheet('metrics')

class MetricRegister():
    def __init__(self):
        self.book = xlwt.Workbook()
        # 创建sheet
        self.sheet = self.book.add_sheet('metrics')
        self.counter = 0


    def __call__(self, gt, prediction):
        indices = np.nonzero(gt!=0)
        ans = self.__measure(prediction[indices], gt[indices])
        recall = ans['class_recall']
        AA = ans['AA']
        OA = ans['OA']
        kappa = ans['kappa']
        i = 0
        while i < len(recall):
            self.sheet.write(self.counter, i, recall[i])
            i += 1
        self.sheet.write(self.counter, i, AA)
        i += 1
        self.sheet.write(self.counter, i, OA)
        i += 1
        self.sheet.write(self.counter, i, kappa)
        self.counter += 1

    def __measure(self, y_pred, y_true):
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.cpu().numpy()
        else:
            y_pred = np.array(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.cpu().numpy()
        else:
            y_true = np.array(y_true)
        # 计算类别 recall 值
        class_recall = recall_score(y_true, y_pred, average=None)
        # 计算平均 recall
        AA = class_recall.mean()
        # 计算准确率
        OA = accuracy_score(y_true, y_pred)
        # 计算 kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        res = {'class_recall': class_recall.tolist(),
               'AA': AA,
               'OA': OA,
               'kappa': kappa}
        return res

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.book.save(f'{save_path}/metrics.xls')
