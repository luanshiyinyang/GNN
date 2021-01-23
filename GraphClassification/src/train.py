import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import DDDataset
from model import ModelA, ModelB

plt.style.use('fivethirtyeight')


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def normalization(adjacency):
    """
    L=D^-0.5 * (A+I) * D^-0.5
    :param adjacency:
    :return:
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


dataset = DDDataset()

# 模型输入数据准备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
adjacency = dataset.sparse_adjacency
normalize_adjacency = normalization(adjacency).to(DEVICE)
node_labels = tensor_from_numpy(dataset.node_labels, DEVICE)
node_features = F.one_hot(node_labels, node_labels.max().item() + 1).float()
graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE)
graph_labels = tensor_from_numpy(dataset.graph_labels, DEVICE)
train_index = tensor_from_numpy(dataset.train_index, DEVICE)
test_index = tensor_from_numpy(dataset.test_index, DEVICE)
train_label = tensor_from_numpy(dataset.train_label, DEVICE)
test_label = tensor_from_numpy(dataset.test_label, DEVICE)

# 超参数设置
INPUT_DIM = node_features.size(1)
NUM_CLASSES = 2
EPOCHS = 200
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001

# 模型初始化
model_g = ModelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
model_h = ModelB(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)


model = model_g
print("Device:", DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.train()
losses_a = []
for epoch in range(EPOCHS):
    logits = model(normalize_adjacency, node_features, graph_indicator)
    loss = criterion(logits[train_index], train_label)  # 只对训练的数据计算损失值
    optimizer.zero_grad()
    loss.backward()  # 反向传播计算参数的梯度
    optimizer.step()  # 使用优化方法进行梯度更新
    train_acc = torch.eq(
        logits[train_index].max(1)[1], train_label).float().mean()
    # print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(epoch, loss.item(), train_acc.item()))
    losses_a.append(loss.item())

model.eval()
with torch.no_grad():
    logits = model(normalize_adjacency, node_features, graph_indicator)
    test_logits = logits[test_index]
    test_acc = torch.eq(
        test_logits.max(1)[1], test_label
    ).float().mean()

print(test_acc.item())

model = model_h
print("Device:", DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.train()
losses_b = []
for epoch in range(EPOCHS):
    logits = model(normalize_adjacency, node_features, graph_indicator)
    loss = criterion(logits[train_index], train_label)  # 只对训练的数据计算损失值
    optimizer.zero_grad()
    loss.backward()  # 反向传播计算参数的梯度
    optimizer.step()  # 使用优化方法进行梯度更新
    train_acc = torch.eq(
        logits[train_index].max(1)[1], train_label).float().mean()
    # print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(epoch, loss.item(), train_acc.item()))
    losses_b.append(loss.item())

model.eval()
with torch.no_grad():
    logits = model(normalize_adjacency, node_features, graph_indicator)
    test_logits = logits[test_index]
    test_acc = torch.eq(
        test_logits.max(1)[1], test_label
    ).float().mean()

print(test_acc.item())


def plot_loss(loss_a, loss_b):
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(loss_a)), loss_a, c=np.array([255, 71, 90]) / 255., label='Global Model')
    plt.plot(range(len(loss_b)), loss_b, c=np.array([120, 80, 90]) / 255., label='Hierarchical Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.title('loss')
    plt.savefig("../assets/loss.png")
    plt.show()


plot_loss(losses_a, losses_b)