import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # [K, 1, N, N]

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]

        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(2)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs)  # [B, N, D]，一个线性层，就是第一步中公式的 W*h

        # 下面这个就是，第i个节点和第j个节点之间的特征做了一个内积，表示它们特征之间的关联强度
        # 再用graph也就是邻接矩阵相乘，因为邻接矩阵用0-1表示，0就表示两个节点之间没有边相连
        # 那么最终结果中的0就表示节点之间没有边相连
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)  # [B, N, D]*[B, D, N]->[B, N, N], x(i)^T * x(j)

        # 由于上面计算的结果中0表示节点之间没关系，所以将这些0换成负无穷大，因为softmax的负无穷大=0
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)  # [B, N, N]，在第２维做归一化，就是说所有有边相连的节点做一个归一化，得到了注意力系数
        return torch.bmm(attention, h) + self.b  # [B, N, N] * [B, N, D]，，这个是第三步的，利用注意力系数对邻域节点进行有区别的信息聚合


class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()
        # 用循环来增加多注意力， 用nn.ModuleList变成一个大的并行的网络
        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])  # in_c为输入特征维度，hid_c为隐藏层特征维度

        # 上面的多头注意力都得到了不一样的结果，使用注意力层给聚合起来
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)

        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        # 每一个注意力头用循环取出来，放入list里，然后在最后一维串联起来
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data, device):
        graph = data["graph"][0].to(device)  # [N, N]
        flow = data["flow_x"]  # [B, N, T, C]
        flow = flow.to(device)  # 将流量数据送入设备

        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)  # [B, N, T * C]
        '''
        上面是将这一段的时间的特征数据摊平做为特征，这种做法实际上忽略了时序上的连续性
        这种做法可行，但是比较粗糙，当然也可以这么做：
        flow[:, :, 0] ... flow[:, :, T-1]   则就有T个[B, N, C]这样的张量，也就是 [B, N, C]*T
        每一个张量都用一个SubNet来表示，则一共有T个SubNet，初始化定义　self.subnet = [GATSubNet(...) for _ in range(T)]
        然后用nn.ModuleList将SubNet分别拎出来处理，参考多头注意力的处理，同理
        '''
        prediction = self.subnet(flow, graph).unsqueeze(2)  # [B, N, 1, C]，这个１加上就表示预测的是未来一个时刻
        return prediction

