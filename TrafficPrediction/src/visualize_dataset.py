import numpy as np
import matplotlib.pyplot as plt

from metrics import MAE, MAPE, RMSE

plt.style.use('fivethirtyeight')


def get_flow(filename):
    flow_data = np.load(filename)
    return flow_data['data']


def show_pred(test_loader, all_y_true, all_predict_values):
    node_id = 166
    plt.title("no. {} node the first day".format(node_id))
    plt.xlabel("time/5min")
    plt.ylabel("flow")
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_y_true)[:24 * 12, node_id, 0, 0], label='true')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[0])[:24 * 12, node_id, 0, 0], label='ChebNet pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[1])[:24 * 12, node_id, 0, 0], label='GCN pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[2])[:24 * 12, node_id, 0, 0], label='GAT pred')
    plt.legend()
    plt.savefig("../assets/the first day pred flow in node {}.png".format(str(node_id)), dpi=400)
    plt.show()

    plt.title("no. {} node the two weeks".format(node_id))
    plt.xlabel("time/5min")
    plt.ylabel("flow")
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_y_true)[:, node_id, 0, 0], label='true')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[0])[:, node_id, 0, 0], label='ChebNet pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[1])[:, node_id, 0, 0], label='GCN pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[2])[:, node_id, 0, 0], label='GAT pred')
    plt.legend()
    plt.savefig("../assets/the first two weeks pred flow in node {}.png".format(str(node_id)), dpi=400)
    plt.show()

    mae = MAE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_y_true),
              test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_predict_values[0]))
    rmse = RMSE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[0]))
    mape = MAPE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[0]))
    print("ChebNet基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))

    mae = MAE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_y_true),
              test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_predict_values[1]))
    rmse = RMSE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[1]))
    mape = MAPE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[1]))
    print("GCN基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))

    mae = MAE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_y_true),
              test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_predict_values[2]))
    rmse = RMSE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[2]))
    mape = MAPE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[2]))
    print("GAT基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))


if __name__ == '__main__':
    traffic_data = get_flow('../dataset/PEMS/PEMS04/data.npz')
    print("data size {}".format(traffic_data.shape))
    # 采样某个结点的数据
    node_id = 224
    plt.plot(traffic_data[: 24 * 12, node_id, 0], label="flow")
    plt.plot(traffic_data[: 24 * 12, node_id, 1], label="speed")
    plt.plot(traffic_data[: 24 * 12, node_id, 2], label="other")
    plt.legend(loc=0)
    plt.savefig("../assets/vis.png")
    plt.show()
