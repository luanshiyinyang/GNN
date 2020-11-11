import numpy as np


def sampling(src_nodes, sample_num, neighbor_table):
    """
    根据源节点一阶采样指定数量的邻居，有放回
    :param src_nodes:
    :param sample_num:
    :param neighbor_table:
    :return:
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        neighbor_nodes = neighbor_table.getrow(sid).nonzero()
        res = np.random.choice(np.array(neighbor_nodes).flatten(), size=sample_num)
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """
    根据源节点进行多阶采样
    :param src_nodes:
    :param sample_nums:
    :param neighbor_table:
    :return:
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result
