import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

from model import GAT_model, GCN
from data_loader import LoadData
from metrics import MAE, MAPE, RMSE

seed = 1001
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["../dataset/PeMS_04/PeMS04.csv", "../dataset/PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = LoadData(data_path=["../dataset/PeMS_04/PeMS04.csv", "../dataset/PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Loading Model
    # TODO:  Construct the GAT (must) and DCRNN (optional) Model

    # my_net = None
    my_net = GAT_model(6, 6, 1)
    # my_net = GCN(6,6,1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Adam_Epoch = 5
    my_net.train()
    for epoch in range(Adam_Epoch):
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

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
                epoch + 1, 10 * epoch_loss / (len(train_data) / 64),
                epoch_mae, epoch_rmse, epoch_mape, (end_time - start_time) / 60))
    SGD_epoch = 15
    optimizer = torch.optim.SGD(my_net.parameters(), lr=0.2, weight_decay=0.005)
    for epoch in range(SGD_epoch):
        if (epoch % 3 == 0):
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

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
                epoch + Adam_Epoch + 1, 10 * epoch_loss / (len(train_data) / 64),
                epoch_mae, epoch_rmse, epoch_mape, (end_time - start_time) / 60))

    my_net.eval()
    with torch.no_grad():
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        total_loss = 0.0
        num = 0
        all_predict_value = 0
        all_y_true = 0
        for data in test_loader:
            predict_value = my_net(data, device).to(torch.device("cpu"))
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
            10 * total_loss / (len(test_data) / 64), epoch_mae, epoch_rmse, epoch_mape))

    # 保存模型
    torch.save(my_net, 'model_GAT_6.pth')

    ####选择节点进行流量可视化
    node_id = 120

    plt.title(str(node_id) + " 号节点交通流量可视化(第一天)")
    plt.xlabel("时刻/5min")
    plt.ylabel("交通流量")
    plt.plot(
        test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true)[:24 * 12, node_id, 0, 0],
        label='真实值')
    plt.plot(
        test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value)[:24 * 12, node_id, 0,
        0], label='预测值')
    plt.legend()
    plt.savefig(str(node_id) + " 号节点交通流量可视化(第一天).png", dpi=400)
    plt.show()

    plt.title(str(node_id) + " 号节点交通流量可视化(两周)")
    plt.xlabel("时刻/5min")
    plt.ylabel("交通流量")
    plt.plot(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true)[:, node_id, 0, 0],
             label='真实值')
    plt.plot(
        test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value)[:, node_id, 0, 0],
        label='预测值')
    plt.legend()
    plt.savefig(str(node_id) + " 号节点交通流量可视化(两周).png", dpi=400)
    plt.show()

    mae = MAE(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true),
              test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value))
    rmse = RMSE(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true),
                test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value))
    mape = MAPE(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true),
                test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value))
    print("基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))


if __name__ == '__main__':
    main()
