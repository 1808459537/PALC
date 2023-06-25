from sklearn.model_selection import KFold
import numpy  as np
import torch
import time
import scipy
import scipy.io as sio
from data.multi_label_data import MultiLabelData
from algorithm.Properties import Properties

class palc:
    """
    Multi-label active learning through serial-parallel networks.
    The main algorithm.
    """

    def __init__(self, para_train_data_matrix, para_test_data_matrix, para_train_label_matrix, para_test_label_matrix,  # 四个矩阵
                 para_num_instances: int = 0, para_num_conditions: int = 0, para_num_labels: int = 0,                   # 矩阵的三个参数
                 para_layer_num_nodes: list = None, para_parallel_layer_num_nodes: list = None,
                 para_learning_rate: float = 0.01, para_mobp: float = 0.6, para_activators: str = "s" * 100):
        # Step 1. Accept parameters.

        # 带入参数 四个矩阵与三个参数, 生成一系列用于存储的矩阵和数组, 并完成一些简单的初始化
        self.dataset = MultiLabelData(para_train_data_matrix=para_train_data_matrix,
                                      para_test_data_matrix=para_test_data_matrix,
                                      para_train_label_matrix=para_train_label_matrix,
                                      para_test_label_matrix=para_test_label_matrix,
                                      para_num_instances=para_num_instances, para_num_conditions=para_num_conditions,
                                      para_num_labels=para_num_labels)

        self.output_file = None
        self.loss=[]
        self.device = torch.device('cpu')

        # 创建网络(创建过程使用GPU), 这个函数中的self.device决定了计算损失函数是否使用GPU
        self.network = tempAnn(self.dataset, para_layer_num_nodes,para_learning_rate, para_activators,)

        self.cold_start_end_time = 0
        self.multi_round_end_time = 0
        self.final_update_end_time = 0


    def bounded_train(self, para_lower_rounds: int = 200, para_checking_rounds: int = 200,
                      para_enhancement_threshold: float = 0.001):

        temp_input, temp_label_matrix = self.dataset.get_train_matrix()
        print("bounded_train")
        for i in range(para_lower_rounds):
            if i % 100 == 0:
                print("round: ", i)
            # 以当前传入的三个矩阵完成对于temp_input的训练,  temp_extended_label_matrix作为目标值
            # 这里的self是创建的masp对象, self.network是masp类旗下的一个MultilabelAnn对象, one_round_train 是 MultilabelAnn类旗下的一个方法
            self.loss=self.network.one_round_train(temp_input,temp_label_matrix)
            print(self.loss)



    def my_test(self):
        my_test_acc, test_f1 = self.network.my_test()
        return my_test_acc, test_f1

def test_full_train(para_dataset_name: str = 'Emotion', para_scheme: int = 0):
    """
    用于填写第一组实验数据, 查询所有标签. 可测试网络的有效性.
    :param para_dataset_name: 数据集名称.
    :param para_scheme: 0 for serial-parallel, 1 for serial only, 2 for parallel only.
    """
    print(para_dataset_name)

    prop = Properties(para_dataset_name, para_scheme)
    if prop.cross_val:
        temp_data_matrix, temp_label_matrix, temp_test_data, temp_test_labels = read_data(para_train_filename=prop.filename)
        temp_acc = np.zeros((prop.cross_validate_num, 1))
        temp_f1 = np.zeros((prop.cross_validate_num, 1))
        kf = KFold(prop.cross_validate_num, shuffle=True)
        for k, (train_index, test_index) in enumerate(kf.split(temp_data_matrix)):
            prop.train_data_matrix = temp_data_matrix[train_index, :]
            prop.test_data_matrix = temp_data_matrix[test_index, :]
            prop.train_label_matrix = temp_label_matrix[train_index, :]
            prop.test_label_matrix = temp_label_matrix[test_index, :]
            prop.num_instances = prop.train_data_matrix.shape[0]
            prop.num_conditions = prop.train_data_matrix.shape[1]
            prop.num_labels = prop.train_label_matrix.shape[1]
            prop.full_connect_layer_num_nodes[0] = prop.num_conditions
            for i in range(len(prop.full_connect_layer_num_nodes) - 1):
                prop.full_connect_layer_num_nodes[i + 1] = 128
            for i in range(len(prop.parallel_layer_num_nodes) - 1):
                prop.parallel_layer_num_nodes[i] = 128
            #prop.full_connect_layer_num_nodes = [prop.num_conditions, 256, 256]
            temp_palc = palc(para_train_data_matrix=prop.train_data_matrix,
                            para_test_data_matrix=prop.test_data_matrix,
                            para_train_label_matrix=prop.train_label_matrix,
                            para_test_label_matrix=prop.test_label_matrix,
                            para_num_instances=prop.num_instances,
                            para_num_conditions=prop.num_conditions,
                            para_num_labels=prop.num_labels,
                            para_full_connect_layer_num_nodes=prop.full_connect_layer_num_nodes,
                            para_parallel_layer_num_nodes=prop.parallel_layer_num_nodes,
                            para_learning_rate=prop.learning_rate,
                            para_mobp=prop.mobp, para_activators=prop.activators)

            temp_palc.dataset.query_all()
            temp_palc.bounded_train(prop.pretrain_rounds, 100, prop.enhancement_threshold)
            temp_acc[k, 0], temp_f1[k, 0] = temp_palc.my_test()
        print(temp_acc)
        print(temp_acc.mean())
        print(temp_acc.std())
        print(temp_f1)
        print(temp_f1.mean())
        print(temp_f1.std())
        sio.savemat(prop.outputfilename, {'acc': temp_acc, 'F1': temp_f1})
    else:
        # 返回了训练矩阵, 训练标签矩阵, 测试矩阵, 测试标签矩阵 (注意, 标签矩阵都经历过转置)
        temp_train_data, temp_train_labels, temp_test_data, temp_test_labels = read_data(para_train_filename=prop.filename, param_cross_flag=False)

        # 把读入的值都赋值给prop对象, 方便管理
        prop.train_data_matrix = temp_train_data
        prop.test_data_matrix = temp_test_data
        prop.train_label_matrix = temp_train_labels
        prop.test_label_matrix = temp_test_labels

        # 读取数据行数
        prop.num_instances = prop.train_data_matrix.shape[0]

        # 条件数目
        prop.num_conditions = prop.train_data_matrix.shape[1]
        # 并确定全连接串行层的输入个数
        prop.full_connect_layer_num_nodes[0] = prop.num_conditions

        # 标签数目
        prop.num_labels = prop.train_label_matrix.shape[1]

        # 创建palc对象
        temp_palc = palc(para_train_data_matrix=prop.train_data_matrix,
                        para_test_data_matrix=prop.test_data_matrix,
                        para_train_label_matrix=prop.train_label_matrix,
                        para_test_label_matrix=prop.test_label_matrix,      # 带入参数: 训练数据集, 测试数据集, 训练标签集, 测试标签集

                        para_num_instances=prop.num_instances,
                        para_num_conditions=prop.num_conditions,
                        para_num_labels=prop.num_labels,                    # 带入参数: 数据行数, 条件属性数, 标签数

                        para_full_connect_layer_num_nodes=prop.full_connect_layer_num_nodes,
                        para_parallel_layer_num_nodes=prop.parallel_layer_num_nodes,

                        para_learning_rate=prop.learning_rate,
                        para_mobp=prop.mobp,
                        para_activators=prop.activators)

        # 采用全查询法, 先对一些标记矩阵进行更新
        temp_palc.dataset.query_all()
        # 训练
        # prop.pretrain_round表示训练轮数
        # 100是checking_rounds, 表示第二次训练的检查点距离
        # 提升阈值是0.001, 用于表示当前训练提升低于这个值时就自动停止
        temp_palc.bounded_train(prop.pretrain_rounds, 100, prop.enhancement_threshold)
        # 新增加的模型测试机制
        temp_acc, temp_f1 = temp_palc.my_test()
        sio.savemat(prop.outputfilename, {'acc': temp_acc, 'F1': temp_f1})


def read_data(para_train_filename: str = "../datasets/DataMat/Emotion.mat", para_test_filename: str = "", param_cross_flag: bool = True):
    # Read mat data.
    if para_test_filename == "":
        temp_all_data = sio.loadmat(para_train_filename)

        # np.array 是生成一个矩阵
        temp_train_data = np.array(temp_all_data['train_data'])  # num_instances * num_conditions
        temp_test_data = np.array(temp_all_data['test_data'])

        # num_labels * num_instances to num_instances * num_labels

        # transpose() 是求矩阵转置
        temp_train_targets = np.array(temp_all_data['train_target']).transpose()
        temp_test_targets = np.array(temp_all_data['test_target']).transpose()
    else:
        # 目前没有导致else执行的情况
        temp_train_all_data = sio.loadmat(para_train_filename)  # Both data and target from train
        temp_test_all_data = sio.loadmat(para_test_filename)  # Both data and target from test

        # squeeze是将多维数据压缩为一维
        temp_train_data = np.squeeze(temp_train_all_data['bags'].tolist())
        temp_test_data = np.squeeze(temp_test_all_data['bags'].tolist())
        temp_train_targets = np.array(temp_train_all_data['targets']).transpose()
        temp_test_targets = np.array(temp_test_all_data['targets']).transpose()
    if param_cross_flag:
        # Stack data and labels.

        # 若采用五折交叉验证, 那么训练集和测试集将共享一个数据集, 因此将训练集与测试集堆叠, 标签也是
        # vstack 是将矩阵纵向堆叠
        # 堆叠后, data_matrix = [ [temp_train_data],
        #                        [temp_test_data] ]
        data_matrix = np.vstack((temp_train_data, temp_test_data))
        label_matrix = np.vstack((temp_train_targets, temp_test_targets))
        # 训练集与测试集叠加

        # ndim 返回矩阵维度
        assert data_matrix.ndim == label_matrix.ndim, 'Dimensional inconsistency!'

        # Normalize data.

        # self.data_matrix = self.data_matrix / self.data_matrix.max(axis=0)
        data_matrix = (data_matrix - data_matrix.min(axis=0)) / \
                      (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 0.0001)
        temp_train_data = data_matrix
        temp_train_targets = label_matrix
        temp_test_data = -1
        temp_test_targets = -1
    else:
        # Normalize data.
        # 归一化
        # self.data_matrix = self.data_matrix / self.data_matrix.max(axis=0)
        # .min(axis=0) 求出每一列的最小值, 构成一个一维数组
        # n维矩阵 - 1维矩阵相当于 被减矩阵的每个维度都减去这个1维矩阵. 若n=2 , 这个过程矩阵的每行都剪去这个1维向量
        # temp_train_data - temp_train_data.min(axis=0) 就是将每个列向量减去本列中最小的那个元素
        temp_train_data = (temp_train_data - temp_train_data.min(axis=0)) / \
                          (temp_train_data.max(axis=0) - temp_train_data.min(axis=0) + 0.0001)
        temp_test_data = (temp_test_data - temp_test_data.min(axis=0)) / \
                         (temp_test_data.max(axis=0) - temp_test_data.min(axis=0) + 0.0001)

        # 这里利用技巧求出了标签矩阵中为1的占比, 比较低, 大概是0.1几
    temp_sum = np.sum(temp_train_targets)
    temp_area_in_train = temp_train_targets.size
    temp_ones_in_train = (temp_sum + temp_area_in_train) / 2
    temp_proportion = temp_ones_in_train / temp_area_in_train

    print("Proportion of 1 in train target (label matrix): ", temp_ones_in_train, " out of ", temp_area_in_train,
          " gets ", temp_proportion)
    return temp_train_data, temp_train_targets, temp_test_data, temp_test_targets


