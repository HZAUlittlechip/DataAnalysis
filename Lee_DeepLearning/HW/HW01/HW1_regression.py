""" 回归DNN """

""" * 导入包 """
# 数值分析
import math
import numpy as np

# 读写数据
import pandas as  pd
import os
import csv

# 显示进度条
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# cuda
from torch.cuda import device

# 绘制学习曲线
from torch.utils.tensorboard import SummaryWriter

""" ———————— 一些函数 ——————————  """
def same_seed(seed):
    """ 设置随机书生成器的种子，可确保实验结果是可以重复的 """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_spilt(data_set, valid_rate, seed):
    """ 利用随机种子划分训练集和测试集 """
    valid_set_size = int(valid_rate *  len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))

    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    """ 用于测试集的验证来对模型进行预测 """
    model.eval() # 将模型设置为评估模式
    preds = [] # 存储预测结果

    for x in tqdm(test_loader): #tqdm加载测试数据
        x = x.to(device)

        with torch.no_grad(): # 防止进行梯度计算会对现有额外修改
            pred = model(x)
            preds.append(pred.detach().cpu()) # 转移结果进入cpu

    preds = torch.cat(preds, dim=0).numpy() # 将预测结果沿0轴拼接且保存为np数组

    return preds

""" ———————— 数据集类 --- 来封装数据集里面的特征和目标让训练模型可以直接使用 ——————————"""
class COVID19Dataset(Dataset):
    def __init__(self, features, target=None):
        if target is None:
            self.target = target
        else:
            self.target = torch.FloatTensor(target) # 转化为能用pytorch能实现的数组
        self.features = torch.FloatTensor(features)

    def __getitem__(self, idx):
        """ 让对象可以用obj[idx]的方式来用返回样本 """
        if self.target is None:
            return self.features[idx]
        else:
            return self.features[idx], self.target[idx]

    def __len__(self):
        return len(self.features)

""" ———————— 神经网络模型 —————————— """
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x) # 让x顺序的进入有序的网络
        x = x.squeeze(1) # 移除张量维度大小上为1的维度

        return x

""" —————————— 特征选取
--- 选取对你模型有易的参数来优化的你的回归模型 ————————————
"""
def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = train_data[:,-1], valid_data[:,-1] # 对于2维数组来说选取最后1列且考虑全部行
    raw_x_train, raw_x_valid, raw_x_test =  \
        train_data[:,:-1] , valid_data[:,:-1], test_data    # [:,:-1]选取除了最后1行的每一行元素

    # 控制feature的选取也是模型优化的手段
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1])) # 选取全部的列数即所有的features
    else:
        feat_idx = [0, 1, 2, 3, 4] # 提取指定的features

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

""" ———————— 超参数调整 --- 配置 ————————— """
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

config = {
    'seed': 5201314,
    'select_all': True,
    'valid_ratio': 0.2, #验证集比例
    'n_epochs' : 3000, # 进行多少轮
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400, # 如果在这么多轮模型都没有改良就停止训练
    'save_path': './models/model.ckpt'
}


""" —————— 数据读取 ———————— """
# 根据设定的种子来可以让实验可以复现
same_seed(config['seed'])

# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./covid.train_new.csv').values, pd.read_csv('./covid.test_un.csv').values
train_data, valid_data = train_valid_spilt(train_data, config['valid_ratio'], config['seed'])

# 打印出数据的尺寸
print(f'train_data size: {train_data.shape}\n'
      f'valid_data size: {valid_data.shape}\n'
      f'test_data size: {test_data.shape}')

# 要素选择
x_train, x_valid, x_test, y_train, y_valid = select_feat(
    train_data, valid_data, test_data, config['select_all']
)

# 输出要素的个数
print(f'number of features: {x_train.shape[1]}')

# 将数据输入到数据集类
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

#利用pytorch将数据打包为batch
""" 
    shuffle = True: 每个epoch训练开始会打乱数据集中的样本顺序，来增加训练的多样性
    pin_memory =True: 启用内存固定机制，将数据夹加载到GPU时会更加高效，可以减少数据从主存到GPU内存的延迟       
"""
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader =DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


""" ——————— 训练过程 —————— """
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # 定义损失函数

    # 优化器是可以调整的比如说
    # 1. 去https://pytorch.org/docs/stable/optim.html去得到更多的可用的算法
    # 2. 可以进行L2正则化在优化器中添加（weight_decay）或者自己去实现，下面的优化器就有权重衰减
    # 权值衰减法
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=0.9, weight_decay=0.01)

    writer = SummaryWriter() # 用来绘制学习曲线来可视化学习过程

    # 检查是否存在一个名为model的，如果没有就创建一个model文件夹来存储模型
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # 初始化参数
    n_epochs, best_loss, step, early_stop_count = \
        config['n_epochs'], math.inf, 0, 0

    # 开始训练
    for epoch in range(n_epochs):
        model.train() #  根据你设定的模型进行训练
        loss_record = []

        # 可视化训练过程（tqdm）
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad() # 重置梯度
            x, y = x.to(device), y.to(device) # 转移数据到GPU上
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward() # 反向传播计算梯度
            optimizer.step() # 更新参数
            step += 1
            # 添加loss的记录且将loss从计算图中分离出
            loss_record.append(loss.detach().item())  #  # （分离出来后loss不再残余梯度计算，后续的操作不会影响到loss计算图上的梯度）

            # 训练轮数和损失函数的可视化显示
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]') # 完成率
            train_pbar.set_postfix({'loss':f'{loss.detach().item():.4f}'}) # 在tqdm精度条后面添加添加损失函数的动态信息

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # 设置模型为验证模型 --- valid_loader
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): # 关闭梯度计算
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # 保存模型，可到上面修改超参数
            print(f'Saving model with loss {best_loss:.3f}...')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

""" ——————— 开始训练 ———————— """
model = My_Model(input_dim=x_valid.shape[1]).to(device) # 把模型放在GPU上跑
trainer(train_loader, valid_loader, model, config, device)

""" ———————— 测试集预测数据  ———————— """
def save_pred(preds, file):
    """ 保存预测的结果 """
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow((['id', 'tested_positive'])) # 写入csv表格的表头
        for i, p in enumerate(preds):
            writer.writerow([i, p]) # 写入预测结果

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) # 进行预测
save_pred(preds, 'pred.csv')






