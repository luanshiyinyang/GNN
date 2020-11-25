import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

from model import GCN, ChebNet, GAT
from metrics import MAE, MAPE, RMSE
from data_loader import get_loader
from visualize_dataset import show_pred

seed = 1001
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Loading Dataset
train_loader, test_loader = get_loader('PEMS08')

gcn = GCN(6, 6, 1)
chebnet = ChebNet(6, 6, 1, 1)
gat = GAT(6, 6, 1, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [chebnet.to(device), gcn.to(device), gat.to(device)]

all_predict_values = []
for i in range(len(models)):
    model = models[i]
    criterion = nn.MSELoss()
    Adam_Epoch = 5
    optimizer = optim.Adam(params=model.parameters())
    SGD_epoch = 15
    #  Train model
    model.train()
    for epoch in range(Adam_Epoch):
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            model.zero_grad()
            predict_value = model(data, device).to(torch.device("cpu"))  # [0, 1] -> recover
            loss = criterion(predict_value, data["flow_y"])
            epoch_mae += MAE(data["flow_y"], predict_value)
            epoch_rmse += RMSE(data["flow_y"], predict_value)
            epoch_mape += MAPE(data["flow_y"], predict_value)

            epoch_loss += loss.item()
            num += 1
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_mae = epoch_mae / num
        epoch_rmse = epoch_rmse / num
        epoch_mape = epoch_mape / num
        print(
            "Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins".format(
                epoch + 1, 10 * epoch_loss / (len(train_loader.dataset) / 64),
                epoch_mae, epoch_rmse, epoch_mape, (end_time - start_time) / 60))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, weight_decay=0.005)
    for epoch in range(SGD_epoch):
        if epoch % 3 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            model.zero_grad()
            predict_value = model(data, device).to(torch.device("cpu"))  # [0, 1] -> recover
            loss = criterion(predict_value, data["flow_y"])
            epoch_mae += MAE(data["flow_y"], predict_value)
            epoch_rmse += RMSE(data["flow_y"], predict_value)
            epoch_mape += MAPE(data["flow_y"], predict_value)

            epoch_loss += loss.item()
            num += 1
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_mae = epoch_mae / num
        epoch_rmse = epoch_rmse / num
        epoch_mape = epoch_mape / num
        print(
            "Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins".format(
                epoch + Adam_Epoch + 1, 10 * epoch_loss / (len(train_loader.dataset) / 64),
                epoch_mae, epoch_rmse, epoch_mape, (end_time - start_time) / 60))

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num = 0
        all_predict_value = 0
        all_y_true = 0
        for data in test_loader:
            predict_value = model(data, device).to(torch.device("cpu"))
            if num == 0:
                all_predict_value = predict_value
                all_y_true = data["flow_y"]
            else:
                all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
                all_y_true = torch.cat([all_y_true, data["flow_y"]], dim=0)
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()
            num += 1
        epoch_mae = MAE(all_y_true, all_predict_value)
        epoch_rmse = RMSE(all_y_true, all_predict_value)
        epoch_mape = MAPE(all_y_true, all_predict_value)
        print("Test Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(
            10 * total_loss / (len(test_loader.dataset) / 64), epoch_mae, epoch_rmse, epoch_mape))

    all_predict_values.append(all_predict_value)
show_pred(test_loader, all_y_true, all_predict_values)


