import time

import numpy as np
import torch
from numpy.core.multiarray import ndarray
from torch import nn
import itertools
from algorithm.Pre_Do import Lap

from data.multi_label_data import MultiLabelData




class ParallelAnn(nn.Module):
    """
    Parallel ANN.

    This class handles the parallel part.
    """

    def __init__(self, para_parallel_layer_num_nodes: list = None, para_activators: str = "s" * 100):
        super().__init__()

        temp_model = []
        for i in range(len(para_parallel_layer_num_nodes) - 1):
            temp_input = para_parallel_layer_num_nodes[i]
            temp_output = para_parallel_layer_num_nodes[i + 1]
            temp_linear = nn.Linear(temp_input, temp_output)
            temp_model.append(temp_linear)
            temp_model.append(get_activator(para_activators[i]))
        self.model = nn.Sequential(*temp_model)

    def forward(self, para_input: torch.tensor = None):
        # temp_input = torch.tensor(para_input, dtype=torch.float).to(self.de)
        # temp_input = para_input.clone().detach()
        # print(para_input)
        temp_output = self.model(para_input)
        return temp_output


class MultiLabelAnn(nn.Module):

    """
    Multi-label ANN.

    This class handles the whole network.
    """
    def __init__(self, para_dataset: MultiLabelData = None, para_full_connect_layer_num_nodes: list = None,
                 para_parallel_layer_num_nodes: list = None, para_learning_rate: float = 0.01,
                 para_mobp: float = 0.6, para_activators: str = "s" * 100, para_device=None):

        super().__init__()
        self.dataset = para_dataset
        self.num_parts = self.dataset.num_labels
        self.num_layers = len(para_full_connect_layer_num_nodes) + len(para_parallel_layer_num_nodes)
        self.learning_rate = para_learning_rate
        self.mobp = para_mobp
        self.device = para_device
        self.skip_count = 0  # For cost-sensitive learning.
        temp_model = []
        for i in range(len(para_full_connect_layer_num_nodes) - 1):
            temp_input = para_full_connect_layer_num_nodes[i]
            temp_output = para_full_connect_layer_num_nodes[i + 1]
            temp_linear = nn.Linear(temp_input, temp_output)
            temp_model.append(temp_linear)
            temp_model.append(get_activator(para_activators[i]))
        self.full_connect_model = nn.Sequential(*temp_model)
        temp_parallel_activators = para_activators[len(para_full_connect_layer_num_nodes) - 1:]

        self.parallel_model = [ParallelAnn(para_parallel_layer_num_nodes, temp_parallel_activators).to(self.device)
                               for _ in range(self.dataset.num_labels)]

        self.my_optimizer = torch.optim.Adam(itertools.chain(self.full_connect_model.parameters(),
                                                             *[model.parameters() for model in self.parallel_model]),
                                             lr=para_learning_rate)

        self.my_loss_function = nn.MSELoss().to(para_device)

    def forward(self, para_input: np.ndarray = None):

        temp_input = torch.as_tensor(para_input, dtype=torch.float).to(self.device)
        temp_inner_output = self.full_connect_model(temp_input)
        #
        # ## softmax特征提取节点
        # softmax_0 = torch.nn.Softmax(dim=0)
        # temp_inner_output = softmax_0(temp_inner_output)

        ##特征合并
        temp_inner_output = torch.cat((temp_inner_output, temp_input), dim=1)

        temp_inner_output = [model(temp_inner_output) for model in self.parallel_model]
        temp_output = temp_inner_output[0]
        for i in range(len(temp_inner_output) - 1):
            temp_output = torch.cat((temp_output, temp_inner_output[i + 1]), -1)

        return temp_output


        # 执行一轮测试

    def one_round_train(self, para_r,para_input: np.ndarray = None,para_extended_label_matrix: np.ndarray = None,para_lap:np.ndarray = None,para_in_lap :np.ndarray= None):
        temp_outputs = self(para_input)
        para_extended_label_matrix = torch.tensor(para_extended_label_matrix, dtype=torch.float)
        #real=torch.tensor(para_really, dtype=torch.float)

        lap=torch.tensor(para_lap, dtype=torch.float)

        in_lap=torch.tensor(para_in_lap, dtype=torch.float)
        outputs=torch.matmul(torch.matmul(temp_outputs,lap), torch.transpose(temp_outputs, dim0=0, dim1=1))
        tr=outputs.trace()/temp_outputs.shape[0]
        out=torch.matmul(torch.matmul(temp_outputs,in_lap), torch.transpose(temp_outputs, dim0=0, dim1=1))
        temp_tr=out.trace()/temp_outputs.shape[0]
        temp_loss = self.my_loss_function(temp_outputs,para_extended_label_matrix) + para_r*tr +0.1* temp_tr

        # 将self.my_optimizer 的梯度设置为0
        self.my_optimizer.zero_grad()
        # 根据损失函数进行bp求惩罚信息
        temp_loss.backward()
        # 更新权值
        self.my_optimizer.step()
        # temp_loss是张量对象, .item()取出了里面的数组, 总的来说,
        # 函数返回了第一次预测的损失函数值, 通过观察发现通过函数使用深入这个值是在不断减少的

        return temp_loss.item()

    def  bounded_train(self,para_r,para_input:np.ndarray = None, para_label:np.ndarray=None,para_lap:np.ndarray = None,para_in_lap :np.ndarray= None):
        for i in range(1000):
            self.loss=self.one_round_train(para_r,para_input,para_label,para_lap,para_in_lap)
            if(i%500==0):
                print(self.loss)
        return np.array(self.loss)

    def test(self):
        temp_input = torch.tensor(self.dataset.test_data_matrix[:], dtype=torch.float, device='cpu')
        temp_predictions = self(temp_input)
        self.dataset.test_predicted_label_matrix = torch.exp(temp_predictions[:, 1::2]) / \
                                                         (torch.exp(temp_predictions[:, 1::2]) + torch.exp(
                                                             temp_predictions[:, ::2]))
        #self.dataset.predicted_label_matrix=temp_switch.int()



def get_activator(para_activator: str = 's'):
    """
    Todo: 对其它激活函数的支持.
    """
    if para_activator == 'r':
        return nn.ReLU()
    elif para_activator == 's':
        return nn.Sigmoid()
    else:
        return nn.Sigmoid()


