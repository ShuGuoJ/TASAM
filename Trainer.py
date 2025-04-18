import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import optimizer as optimizer_
from sklearn.metrics import roc_auc_score


class SpectralTrainer(object):
    r"""模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 模型训练
    def train(self, loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device, monitor=None):
        self.model.train()
        self.model.to(device)
        criterion.to(device)
        losses = []
        for x, x1, x2, x_gabor, y in loader:
            # 数据预处理
            batch, patch = x.shape[:2]
            x_gabor = x_gabor.reshape((batch, patch ** 2, -1)).permute(2, 0, 1)
            x = x.permute(0, 3, 1, 2)
            x1 = x1.permute(0, 3, 1, 2)
            x2 = x2.permute(0, 3, 1, 2)
            # 将数据加载到GPU
            x1, x2 = x1.to(device), x2.to(device)
            x,x_gabor,y = x.to(device),x_gabor.to(device), y.to(device)-1
            # 前向传播
            logits = self.model(x, x_gabor, x1, x2)
            # 反向传播
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if monitor is not None:
                monitor.add([self.get_parameters()], ord=2)
            return float(np.mean(losses))

    # 模型验证
    def evaluate(self, loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        self.model.to(device)
        losses = []
        preds = []
        truth_label = []
        with torch.no_grad():
            for x, x1, x2, x_gabor, y in loader:
                # 数据预处理
                batch, patch = x.shape[:2]
                x_gabor = x_gabor.reshape((batch, patch ** 2, -1)).permute(2, 0, 1)
                x = x.permute(0, 3, 1, 2)
                x1 = x1.permute(0, 3, 1, 2)
                x2 = x2.permute(0, 3, 1, 2)
                # 将数据加载到GPU
                x1, x2 = x1.to(device), x2.to(device)
                x, x_gabor, y = x.to(device), x_gabor.to(device), y.to(device)-1
                logits = self.model(x, x_gabor, x1, x2)
                truth_label.append(y.clone().detach().cpu())
                preds.append(torch.softmax(logits.clone().detach().cpu(), dim=-1).argmax(dim=-1))
                loss = criterion(logits, y)
                losses.append(loss.cpu().numpy())
        # 计算准确率
        truth_label = torch.cat(truth_label, dim=0)
        preds = torch.cat(preds, dim=0)
        acc = torch.sum(truth_label==preds) / truth_label.numel()
        return np.mean(losses), acc

    # 模型测试
    def predict(self, loader, device: torch.device):
        self.model.eval()
        self.model.to(device)
        preds = []
        probs = []
        with torch.no_grad():
            for x, x1, x2, x_gabor, y in loader:
                # 数据预处理
                batch, patch = x.shape[:2]
                x_gabor = x_gabor.reshape((batch, patch ** 2, -1)).permute(2, 0, 1)
                x_gabor = x_gabor.to(device)
                x = x.permute(0, 3, 1, 2)
                x1 = x1.permute(0, 3, 1, 2)
                x2 = x2.permute(0, 3, 1, 2)
                # 将数据加载到GPU
                x, x1, x2 = x.to(device) , x1.to(device), x2.to(device)
                logits = self.model(x, x_gabor, x1, x2)
                pred = logits.argmax(-1) + 1
                prob = torch.softmax(logits.clone().detach().cpu(), dim=-1)
                preds.append(pred.clone().detach().cpu())
                probs.append(prob)
        return torch.cat(preds, dim=0), torch.cat(probs, dim=0)

    def get_parameters(self):
        return self.model.parameters()

    def save(self, path):
        torch.save(self.model.cpu().state_dict(), path)
