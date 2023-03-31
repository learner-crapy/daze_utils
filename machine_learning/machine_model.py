import os
import random
# 获取当前文件夹路径
import pandas as pd
import openpyxl
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 线性回归获取参数
def LinReg_w_b(reg):
    # 获取模型的偏置项和权重
    b = reg.intercept_
    w = reg.coef_

    return w, b

# 线性回归训练模型

def TrainLinReg(x, y):
    # 准备训练数据
    # x = np.array([[1], [2], [3], [4], [5]])
    # y = np.array([1, 2, 3, 4, 5])

    # 创建线性回归模型
    reg = LinearRegression().fit(x, y)

    # 使用模型预测
    # y_pred = reg.predict([[6]])

    return reg


# 线性回归模型评价
def ScoreLinReg(reg, x_test, y_test):
    '''
    均方误差（Mean Squared Error，MSE）：表示预测误差的平均值，越小说明模型越好。

    平均绝对误差（Mean Absolute Error，MAE）：表示预测误差的绝对值的平均值，越小说明模型越好。

    R平方（R-squared）：表示模型的预测能力，值越接近1说明模型越好。

    '''
    y_pred = reg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2


# 定义决策树方法进行预测
def DecisionTree(X_train, y_train):


    # 加载数据集
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target

    # 分割训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建决策树分类器并训练
    clf = DecisionTreeRegressor(random_state=42)
    clf.fit(X_train, y_train)

    # 使用测试集评估模型
#     accuracy = clf.score(X_test, y_test)
#     print("Accuracy:", accuracy)

    # 预测一个新样本
#     new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
#     prediction = clf.predict(new_sample)
#     print("Prediction:", prediction)
    return clf

# 决策树评估模型
def ScoreCLF(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    return accuracy
    
    

# 定义一个函数，使用支持向量机进行预测

    '''
    SVR 可以使用多种内核，其中包括线性内核、多项式内核和径向基函数内核。

    线性内核（linear）：线性内核是 SVR 中最常用的内核，它把数据拟合为一条直线。
    regressor = SVR(kernel='linear')


    多项式内核（poly）：多项式内核在线性内核的基础上，引入了更高阶的特征，使模型能够捕捉到数据中的曲线关系。
    regressor = SVR(kernel='poly', degree=3, coef0=1)


    径向基函数内核（rbf）：径向基函数内核是 SVR 中另一种常用的内核，它在数据的每个点处构建了一个径向基函数，以形成对数据的拟合。regressor = SVR(kernel='poly', degree=3, coef0=1)
    regressor = SVR(kernel='rbf', gamma='scale')，其中，gamma 参数表示内核的宽度。


    使用不同的内核会对 SVR 的拟合效果产生重要影响，因此，在选择内核时应该仔细考虑数据的特征。
    '''
def SVRlinear(X_train, y_train):
    regressor = SVR(kernel='linear')
    regressor.fit(X_train, y_train)
    return regressor

def SVRpoly(X_train, y_train):
    regressor = SVR(kernel='poly', degree=3, coef0=1)
    regressor.fit(X_train, y_train)
    return regressor

def SVRrbf(X_train, y_train):
    regressor = SVR(kernel='rbf', gamma='scale')
    regressor.fit(X_train, y_train)
    return regressor

# 评估  
def SVRScore(regressor, x_test, y_test):
    y_pred = regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2
