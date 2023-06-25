import numpy as np
class Lap:
    def __init__(self,lable_real_matrix:np.ndarray=None):
        self.lable_matrix=lable_real_matrix
        self.lable_correlation_Martrix=np.zeros((self.lable_matrix.shape[1],self.lable_matrix.shape[1]))
        self.Laplacian_matrix_D = np.zeros((self.lable_matrix.shape[1], self.lable_matrix.shape[1]))

        self.f1_matrix=np.zeros((self.lable_matrix.shape[1],self.lable_matrix.shape[1]))
        self.target=0

        self.extend_Laplacian_matrix_row=np.zeros((self.lable_matrix.shape[1]*2, self.lable_matrix.shape[1]))
        self.extend_Laplacian_matrix=np.zeros((self.lable_matrix.shape[1]*2, self.lable_matrix.shape[1]*2))


        self.in_lable_correlation_Martrix=np.eye(self.lable_matrix.shape[1]*2)#np.zeros((self.lable_matrix.shape[1]*2, self.lable_matrix.shape[1]*2))


        self.L=np.zeros((self.lable_matrix.shape[1], self.lable_matrix.shape[1]))
        # self.a=self.lable_matrix[:,0]
        # self.b=self.lable_matrix[:,0]
    def compute_Cosin(self,a,b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm==0.0 or b_norm==0.0:
            cos = a_norm*b_norm
        else:
            cos = np.dot(a, b) / (a_norm * b_norm)
        return cos

    def compute_f1(self,a,b):
        c=0.0#1
        d=0.0#0.5
        e=0.0#-0.2
        for i in range(len(a)):
                if(a[i]==1 and b[i]==1):
                    c=c+1
                elif(a[i]==0 and b[i]==0):
                    d=d+0.5
                else:
                    e=e-0.2

        return (c+d+e)/len(a)



    # def compute_f1(self,a,b):
    #     temp = 0
    #     same = 0
    #
    #     for i in range(len(a)):
    #             if(a[i]==1 or b[i]==1):
    #                 temp=temp+1
    #                 if(a[i]==b[i]):
    #                     same=same+1
    #     if(temp==0):
    #         same==0
    #         temp=0.0001
    #     return same/temp

    def compute_F1(self):
        for i in range(self.lable_matrix.shape[1]):
            for j in range(self.lable_matrix.shape[1]):
                if(i!=j):
                    self.f1_matrix[i][j]=self.compute_f1(self.lable_matrix[:,i],self.lable_matrix[:,j])
        return self.f1_matrix

    #余弦
    # def correlation_Martrix(self,):
    #     for i in range(self.lable_matrix.shape[1]):
    #         for j in range(self.lable_matrix.shape[1]):
    #             if(i!=j):
    #                 self.lable_correlation_Martrix[i][j]=self.compute_Cosin(self.lable_matrix[:,i],self.lable_matrix[:,j])
    #     return self.lable_correlation_Martrix



    def correlation_Martrix(self):
        for i in range(self.lable_matrix.shape[1]):
            for j in range(self.lable_matrix.shape[1]):
                if(i!=j):
                    self.lable_correlation_Martrix[i][j]=self.compute_f1(self.lable_matrix[:,i],self.lable_matrix[:,j])
        return self.lable_correlation_Martrix


    def in_correlation_Martrix(self):
        for i in range(self.lable_matrix.shape[1]*2):
            for j in range(self.lable_matrix.shape[1]*2):
                if(i%2==0 and i-j==-1):
                    self.in_lable_correlation_Martrix[i][j]=-1
                if(i%2!=0 and i-j==1):
                    self.in_lable_correlation_Martrix[i][j]=-1

        # for i in range(self.lable_matrix.shape[1] * 2):
        #     for j in range(self.lable_matrix.shape[1] * 2):
        #         if(self.in_lable_correlation_Martrix[i][j]==1):
        #             self.in_lable_correlation_Martrix[i][j]=-1
        #         if (self.in_lable_correlation_Martrix[i][j] == -1):
        #             self.in_lable_correlation_Martrix[i][j] = 1
        return self.in_lable_correlation_Martrix



    def correlation_Martrix_by02(self):
        for i in range(self.lable_matrix.shape[1]):
            for j in range(self.lable_matrix.shape[1]):
                if(i!=j):
                    self.lable_correlation_Martrix[i][j]=self.compute_distance(self.lable_matrix[:,i],self.lable_matrix[:,j])
        return self.lable_correlation_Martrix

    def Laplacian_matrix(self):
        for i in range(self.lable_matrix.shape[1]):
            for j in range(self.lable_matrix.shape[1]):
                if(i==j):
                    self.Laplacian_matrix_D[i][j]=self.lable_correlation_Martrix[i].sum()
        self.L=self.Laplacian_matrix_D-self.lable_correlation_Martrix
        return self.Laplacian_matrix_D-self.lable_correlation_Martrix

    def extended_Laplacian_matrix_rowcompute(self):
        for i in range(self.lable_matrix.shape[1]):
            self.extend_Laplacian_matrix_row[i*2]=self.L[i]


    def  extended_Laplacian_matrix_line(self):
        self.extended_Laplacian_matrix_rowcompute()
        for i in range(self.lable_matrix.shape[1]):
            self.extend_Laplacian_matrix[:,i*2]=self.extend_Laplacian_matrix_row[:,i]
        return self.extend_Laplacian_matrix



    def Get_target(self):

        self.target=np.dot( np.dot(self.lable_matrix,self.Laplacian_matrix()),self.lable_matrix.T)
            ##print(np.dot( np.dot(self.lable_matrix[i].T,self.Laplacian_matrix()),self.lable_matrix[i]))
        return self.target.trace()/self.lable_matrix.shape[0]

    def test(self):
        return  np.dot( np.dot(self.lable_matrix[4].T,self.Laplacian_matrix()),self.lable_matrix[4])