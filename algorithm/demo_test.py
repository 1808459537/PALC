import scipy
import scipy.io as sio
import numpy  as np
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

prop = Properties('')
# prop.num_instances = prop.train_data_matrix.shape[0]
#
# # 条件数目

temp_data_matrix, temp_label_matrix, temp_test_data, temp_test_labels = read_data(para_train_filename=prop.filename,param_cross_flag=False)
dataset=MultiLabelData(temp_data_matrix, temp_test_data,temp_label_matrix, temp_test_labels,para_num_labels=temp_label_matrix.shape[1],para_num_instances=temp_label_matrix.shape[0])


# 条件数目
prop.num_conditions = temp_data_matrix.shape[1]
# 并确定全连接串行层的输入个数
prop.full_connect_layer_num_nodes[0] = prop.num_conditions
lap=Lap(dataset.label_matrix)
lap.correlation_Martrix()

print("这是分界线")

lap.in_correlation_Martrix()


print(lap.Get_target())
lap.extended_Laplacian_matrix_line()



#
# work=All_ann(dataset,prop.layer_num_nodes,prop.learning_rate,prop.activators)
# work.bounded_train(temp_data_matrix,temp_label_matrix)

work=MultiLabelAnn(dataset,prop.full_connect_layer_num_nodes,prop.parallel_layer_num_nodes,prop.learning_rate,prop.activators)

work.bounded_train(prop.para_r,temp_data_matrix,dataset.extended_label_matrix,lap.extended_Laplacian_matrix_line(),lap.in_correlation_Martrix())
# temp_predictions = work.forward(temp_test_data[:])
# temp_switch = temp_predictions[:,::2] < temp_predictions[:,1::2]
#
# dataset.test_predicted_label_matrix= temp_switch.int()
work.test()
# tempSwitch=dataset.test_predicted_label_matrix[:,:]>0.5
# dataset.test_predicted_label_matrix=tempSwitch.int()
# print(dataset.test_predicted_label_matrix)
dataset.compute_auc()
print(dataset.compute_f1())
print(dataset.computeNDCG())

# work=tempAnn(dataset,prop.layer_num_nodes,prop.learning_rate,prop.activators)
# for i in range(2000):
#     print(work.one_round_train(temp_data_matrix,temp_label_matrix))
# work.one_round_train(temp_data_matrix,temp_label_matrix)
#     print(work.net[0].one_round_train(temp_data_matrix,temp_label_matrix))
# work.net[0].one_round_train(temp_data_matrix,temp_label_matrix)