import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from sklearn import metrics



class MultiLabelData:
    """
    Multi-label data.

    This class handles the whole data.
    """

    def __init__(self, para_train_data_matrix, para_test_data_matrix, para_train_label_matrix, para_test_label_matrix,
                 para_num_instances: int = 0, para_num_conditions: int = 0, para_num_labels: int = 0):
        """
        Construct the dataset.
        :param para_train_filename: The training filename.
        :param para_test_filename: The testing filename. The testing data are not employed for testing.
            They are stacked to the training data to form the whole data.
        :param para_num_instances:
        :param para_num_conditions:
        :param para_num_labels:
        """
        # Step 1. Accept parameters.
        self.num_instances = para_num_instances
        self.num_conditions = para_num_conditions
        self.num_labels = para_num_labels

        self.data_matrix = para_train_data_matrix
        self.label_matrix = para_train_label_matrix
        self.test_data_matrix = para_test_data_matrix
        self.test_label_matrix = para_test_label_matrix

        # 将标签矩阵中的-1转变为0
        self.test_label_matrix[self.test_label_matrix == -1] = 0
        self.label_matrix[self.label_matrix == -1] = 0  # -1 to 0


        # .reshape(-1) 可以把测试用的标签矩阵转变为(n*l)*1的列向量
        # 这个数组的作用是在计算F1时用来作为"被"排序对象, 作为真实值的代表
        self.test_label_matrix_to_vector = self.test_label_matrix.reshape(-1) # test label matrix n*l to vector

        # 计算预测标签矩阵中每个标签为正的概率, 用于对test_label_matrix_to_vector进行排序用的
        self.test_predicted_proba_label_matrix = np.zeros((self.num_instances, self.num_labels))

        # 对整个训练集数据进行预测得到的矩阵(双端口已经变为单端口, 1/0 )
        self.predicted_label_matrix = np.zeros((self.num_instances, self.num_labels))
        # 对整个测试集数据进行预测得到的矩阵(双端口已经变为单端口, 1/0 )
        self.test_predicted_label_matrix = np.zeros(self.test_label_matrix.size)

        self.data_matrix_torch = torch.tensor(self.data_matrix, dtype=torch.float)

        self.extended_label_matrix = np.zeros((self.num_instances, self.num_labels * 2))
        for i in range(self.num_instances):
            for j in range(self.num_labels):
                if self.label_matrix[i][j] == 0:
                    self.extended_label_matrix[i][j * 2] = 1
                    self.extended_label_matrix[i][j * 2 + 1] = 0
                else:
                    self.extended_label_matrix[i][j * 2] = 0
                    self.extended_label_matrix[i][j * 2 + 1] = 1

    def __str__(self):
        return str(self.data_matrix) + "\r\n" + str(self.label_matrix)

    def reset(self):
        """
        Reset variables in learning.
        """
        # 已预测标签矩阵 n * l 归-1
        self.predicted_label_matrix.fill(-1)

    def compute_training_accuracy(self):
        """
        Compute the training accuracy using only known labels.
        :param para_output: The predicted label matrix
        :return: The accuracy.
        """
        temp_correct = 0.0
        # 避免除0 (全缺失)
        temp_total_query = 0.001
        for i in range(self.num_instances):
            for j in range(self.num_labels):
                # 若这个标签不缺失
                # 而且只有self.label_query_matrix[i][j] = 1的标签才能被训练, 这个是训练的前提
                if self.label_query_matrix[i][j] == 1:
                    temp_total_query += 1
                    if self.predicted_label_matrix[i][j] == self.label_matrix[i][j]:
                        temp_correct += 1

        print("temp_correct = ", temp_correct, ", temp_total_query = ", temp_total_query)
        return temp_correct / temp_total_query

    def compute_temp_testing_accuracy(self):
        """
        Compute the testing accuracy using unknown labels on the training set (not the testing set).
        :param para_output: The predicted label matrix
        :return: The accuracy.
        """
        temp_correct = 0.0
        # 避免除0 (完全不缺失, 都参与了训练)
        temp_total = 0.001 # Otherwise it might be 1
        for i in range(self.num_instances):
            for j in range(self.num_labels):
                # 只有那些标记为缺失的标签才没有参与训练, 这些标签就是我们要测试的对象
                if self.label_query_matrix[i][j] == 0:
                    temp_total += 1
                    # 虽然定义为缺失了, 但是这个标签本身在label_matrix中可能是有实际含义的, 这种缺失只是主动学习中的尚未查询导致的缺失
                    # 通过训练得到的网络可对于这些尚未查询的逻辑缺失数据进行预测并且估计准确度
                    if self.predicted_label_matrix[i][j] == self.label_matrix[i][j]:
                        temp_correct += 1

        print("num_instances = ", self.num_instances, ", num_labels = ", self.num_labels)
        print("temp_correct in testing = ", temp_correct, ", temp_total in testing = ", temp_total)
        return temp_correct / temp_total

    def compute_testing_f1(self):
        """
        Compute the F1 on the tseting set.
        :return: The F1.
        """
        TN, FP, FN, TP = confusion_matrix(self.test_label_matrix.reshape(-1), self.test_predicted_label_matrix.cpu().numpy().reshape(-1)).ravel()

        print("confusion matrix on the testing set is: ****************************")
        print("TN: ", TN, "FP: ", FP, "FN: ", FN, "TP: ", TP)
        print("Acc confusion matrix:", (TP + TN)/self.test_label_matrix.size)
        print("confusion matrix end ***********************************************")

        return 2*TP / (self.test_label_matrix.size + TP - TN)

    # 计算测试集的标签矩阵预测中 正确个数, 正负样本个数. 并返回预测正确的准确度
    def compute_my_testing_accuracy(self):
        """

        """
        # 获得测试标签矩阵的 n 与 l
        temp_test_num_instances, temp_test_num_labels = self.test_label_matrix.shape

        # .cpu().numpy() 是将张量转换为 numpy对象, 因为比较的对象self.test_label_matrix是一个numpy
        # 得到True/False矩阵, 并求和, 得知识别正确的数目
        temp_correct = np.sum(self.test_predicted_label_matrix.cpu().numpy() == self.test_label_matrix)

        # 统计预测标签的正样本和负样本个数
        temp_positive = np.sum(self.test_predicted_label_matrix.cpu().numpy())
        temp_negative = temp_test_num_instances * temp_test_num_labels - temp_positive

        print("positive, negative predictions are: ", temp_positive, temp_negative)

        return temp_correct / temp_test_num_instances / temp_test_num_labels

    def get_train_matrix(self):
        return self.data_matrix,self.label_matrix

    def compute_f1(self):
        """
           our F1
        """
        # self.test_predicted_proba_label_matrix 是 CUDA的tensor, 先用cpu()将其转换为cpu float-tensor, 随后再转到numpy格式
        # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
        # .detach()可以用于消除梯度, requires_grad为false, 得到的这个tensor永远不需要计算其梯度, 不具有grad
        # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
        temp_proba_matrix_to_vector = self.test_predicted_label_matrix.reshape(-1).cpu().detach().numpy()

        # np.argsort 可以获得排序下标
        # 例[5,12,7]
        # 返回:
        # 例[0,2,1]
        temp = np.argsort(-temp_proba_matrix_to_vector)

        # self.test_label_matrix_to_vector 是由测试用的标签矩阵转变为的1*(n*l)数组, 是个numpy对象
        # self.test_label_matrix_to_vector -> 表征全部目标标签的一维数组, 值是{1, 0}
        # temp_proba_matrix_to_vector - > 是由网络跑出来的预测值softmax概率单端口矩阵转变为的 1*(n*l)数组, 是个numpy对象
        # temp_proba_matrix_to_vector - > 表征全部预测标签的一维数组, 值是[0~1]

        # 按照预测标签为正标签的概率从大到小重排 目标标签一维数组
        # 得到的目标标签一维重排数组一定是 1 1 1 1 1 ... 1 1 1 0 0 0 ... 0 0 0
        # 这里的值本身就是实际的标签值
        # 我们假定都将其预测为正标签, 而排序的目的是更大地确保在最开始前面大部分都能预测正确
        # 这样最开始的实际1个数就能更多与我们预测的正标签匹配, 从而逼近F1的best曲线
        # 自然, 当这个111中没有夹杂0, 0000中没有夹杂1的时候就是完美的F1, 反着推导回去也就证明:
        # 我们双端口的softmax概率值中一定能找到一个阈值, 大于这个阈值的双端口为正标签, 小于这个阈值为负标签, 如此之后能保证预测100%正确
        # 即存在一个完美的分水岭, 分水岭左右没有杂质

        all_label_sort = self.test_label_matrix_to_vector[temp]
        temp_y_F1 = np.zeros(temp.size)

        # all_TP 表示真实为正的个数(TP + FN)
        all_TP = np.sum(self.test_label_matrix_to_vector == 1)

        # 在[0 ~ temp.size] 范围内枚举F1, 每一例都按照正标签去预测
        for i in range(temp.size):
            # 变量TP为范围内为1的个数, 也就是当前范围内真的为1的个数, 即TP
            # i + 1 表示预测为正的次数, 因为整个for循环是预测为正的for循环, 因此i+1就是TP + FP
            TP = np.sum(all_label_sort[0:i+1] == 1)
            P = TP / (i+1)
            R = TP / all_TP
            if (P+R)==0:
                temp_y_F1[i] = 0
            else:
                temp_y_F1[i] = 2.0*P*R / (P+R)
        print("compute f_1:",np.max(temp_y_F1) )
        return np.max(temp_y_F1)


    def computeAUC(self):
        '''
        Compute the AUC

        :return: The AUC
        '''
        tempPredictMatrix = self.test_predicted_label_matrix.detach().numpy()
        tempTargetMatrix = self.test_label_matrix


        auc = metrics.roc_auc_score(tempTargetMatrix, tempPredictMatrix)
        print("computeAUC:", auc)
        return auc


    def compute_auc(self):
        temp_predict_vector = self.test_predicted_label_matrix.detach().numpy().reshape(-1)
        temp_test_target_vector = self.test_label_matrix.reshape(-1)
        temp_predict_sort_index = np.argsort(temp_predict_vector)

        M, N = 0, 0
        for i in range(temp_predict_vector.size):
            if temp_test_target_vector[i] == 1:
                M += 1
            else:
                N = N + 1
                pass
            pass

        sigma = 0
        for i in range(temp_predict_vector.size - 1, -1, -1):
            if temp_test_target_vector[temp_predict_sort_index[i]] == 1:
                sigma += i + 1
                pass
            pass
        auc = (sigma - (M + 1) * M / 2) / (M * N)
        print("compute_auc:", auc)
        return auc


    def computeNDCG(self):

        # 获得概率序列与原目标序列
        tempProbVector = self.test_predicted_label_matrix.detach().numpy().reshape(-1)
        tempTargetVector = self.test_label_matrix.reshape(-1)

        # 按照概率序列排序原1/0串
        temp = np.argsort(-tempProbVector)
        allLabelSort = tempTargetVector[temp]

        # 获得最佳序列: 1111...10000...0
        sortedTargetVector = np.sort(tempTargetVector)[::-1]

        # compute DCG(使用预测的顺序, rel是真实顺序, 实际是111110111101110000001000100
        DCG = 0
        for i in range(temp.size):
            rel = allLabelSort[i]
            denominator = np.log2(i + 2)
            DCG += (rel / denominator)

        # compute iDCG(使用最佳顺序: 11111111110000000000)
        iDCG = 0
        for i in range(temp.size):
            rel = sortedTargetVector[i]
            denominator = np.log2(i + 2)
            iDCG += (rel / denominator)

        ndcg = DCG / iDCG
        print("computeNDCG: ", ndcg)
        return ndcg




