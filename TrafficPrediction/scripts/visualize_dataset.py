import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def get_flow(filename):
    flow_data = np.load(filename)
    return flow_data['data']


if __name__ == '__main__':
    traffic_data = get_flow('../dataset/PeMS_04/PeMS04.npz')
    print("data size {}".format(traffic_data.shape))
    # 采样某个结点的数据
    node_id = 224
    plt.plot(traffic_data[: 24 * 12, node_id, 0], label="flow")
    plt.plot(traffic_data[: 24 * 12, node_id, 1], label="speed")
    plt.plot(traffic_data[: 24 * 12, node_id, 2], label="other")
    plt.legend(loc=0)
    plt.show()
