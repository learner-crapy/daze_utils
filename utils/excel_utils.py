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

# 定义一个函数取出某个文件的某几行
def GetColumns(FileName, SheetName, ColumnList):
    # 读取excel文件
    df = pd.read_excel(FileName, SheetName)

    # 取出"column1"和"column2"列的数据
    selected_columns = df[ColumnList]

    # 返回
    return selected_columns


# 定义一个函数判断表格中有多少个sheet

def SheetNames(FileName):
    # Load the workbook
    workbook = openpyxl.load_workbook(FileName)

    # Get the number of sheets
    SheetNames = workbook.sheetnames

    return SheetNames


# 定义一个函数，返回某个excel中某个sheet的内容

def GetSheetContent(FileName, SheetName):
    # 读取 Excel 文件
    df = pd.read_excel(FileName, sheet_name=SheetName)

    return df


# 定义一个获取当前目录下的所有后缀为xlsx的文件
def GetFileXlsx():
    current_directory = os.getcwd()

    # 创建空列表，用于存储所有.xlsx文件的文件名
    xlsx_files = []


    # 遍历当前文件夹中的所有文件
    for filename in os.listdir(current_directory):
        # 如果文件名以.xlsx结尾
        if filename.endswith(".xlsx"):
            # 将文件名添加到xlsx_files列表中
            xlsx_files.append(filename)

    return xlsx_files


# 定义一个函数，获取一个表格的所有列名
def GetColumnNames(FileName, SheetName):
    df = pd.read_excel(FileName, SheetName)
    return df.columns


