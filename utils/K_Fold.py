class K_Fold:
    def __init__(self, x_y, k):
        self.train_sample = []  # 用于存放训练样本
        self.test_sample = []  # 用于存放测试样本
        self.x_y = x_y
        self.k = k
        self.Random()
        self.Split()

    # 将原来的顺序打乱重新排列
    def Random(self):
        # 随机生成x_y.shape[0]个范围是[0,x_y.shape[0]]的随机自然整数
        # print(self.x_y.shape[0])
        index = list(range(0, self.x_y.shape[0]))
        random.shuffle(index)
        # print(index)
        # 便利矩阵的每一行，每一行按照对应的坐标顺序放到该位置
        x_y_bak = self.x_y
        for i in range(0, self.x_y.shape[0]):
            # print(self.x_y[index[i]:index[i]+1].shape)
            x_y_bak[i:i + 1] = self.x_y[index[i]:index[i] + 1]
        return x_y_bak

    # 进行训练集和测试集的划分
    def Split(self):
        loc = 0  # 开始的位置
        for i in range(0, self.k):
            # 取出测试样本，其余的做训练样本
            test = self.x_y[loc:loc + round(self.x_y.shape[0] / self.k)]
            self.test_sample.append(test)
            # 其余的使用矩阵拼接法拼接起来 
            train = np.concatenate((self.x_y[0:loc], self.x_y[loc + round(self.x_y.shape[0] / self.k):]), axis=0)
            self.train_sample.append(train)
            loc += round(self.x_y.shape[0] / self.k)
