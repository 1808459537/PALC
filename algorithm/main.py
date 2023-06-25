import scipy
import scipy.io as sio
import numpy  as np
from sklearn.model_selection import KFold

from algorithm.Properties import Properties
from data.multi_label_data import MultiLabelData

from algorithm.Pre_Do import Lap



from algorithm.ann.ann import ParallelAnn,MultiLabelAnn

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

        data_matrix = np.vstack((temp_train_data, temp_test_data))
        label_matrix = np.vstack((temp_train_targets, temp_test_targets))
        # 训练集与测试集叠加

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

prop = Properties('Flags')
f1 =[]
ndcg = []
auc = []
temp_data_matrix, temp_label_matrix, temp_test_data, temp_test_labels = read_data(para_train_filename=prop.filename,param_cross_flag=True)
temp_f1= np.zeros((prop.cross_validate_num, 1))
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
            # for i in range(len(prop.full_connect_layer_num_nodes) - 1):
            #     prop.full_connect_layer_num_nodes[i + 1] = 128
            for i in range(len(prop.parallel_layer_num_nodes) - 1):
                prop.parallel_layer_num_nodes[i] = prop.full_connect_layer_num_nodes[-1] + prop.train_data_matrix.shape[1]


            dataset = MultiLabelData(prop.train_data_matrix, prop.test_data_matrix, prop.train_label_matrix,
                                     prop.test_label_matrix, para_num_labels=prop.train_label_matrix.shape[1],
                                     para_num_instances=prop.train_label_matrix.shape[0])

            lap = Lap(prop.train_label_matrix)
            lap.correlation_Martrix()
            print(lap.Get_target())
            # lap.extended_Laplacian_matrix_line()
            work = MultiLabelAnn(dataset, prop.full_connect_layer_num_nodes, prop.parallel_layer_num_nodes,
                                 prop.learning_rate, prop.activators)
            work.bounded_train(prop.para_r, prop.train_data_matrix, dataset.extended_label_matrix,
                               lap.extended_Laplacian_matrix_line(), lap.in_correlation_Martrix())
            work.test()

            auc.append(dataset.compute_auc())
            f1.append(dataset.compute_f1())
            ndcg.append(dataset.computeNDCG())

if __name__ == '__main__':

    print("f1:", np.mean(f1),"+",np.std(f1))
    print("ndcg:", np.mean(ndcg),"+",np.std(ndcg))
    print("auc:", np.mean(auc),"+",np.std(auc))