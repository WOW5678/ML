# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/13 0013 上午 10:00
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 机器学习实战案例：普通最小二乘法求线性回归
"""
#线性回归的类
from sklearn.linear_model import LinearRegression
#原始数据=训练数据+测试数据 数据划分的类
from sklearn.model_selection import train_test_split
#数据标准化
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
import time

#设置字符集，防止中文乱码
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']='SimHei'
mpl.rcParams['axes.unicode_minus']=False

#加载数据
## 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
path='data/household_power_consumption_1000.txt'
df=pd.read_csv(path,sep=';',low_memory=False)
#没有混合类型的时候可以通过low_memory=FALSE调用更多的内存，加快效率

# print(df.head(2))
# print(df.index)
# print(df.columns)
# #查看数据结构
# print(df.info())

#异常数据处理(异常数据过滤)
#替换非法字符为np.nan
new_df=df.replace('?',np.nan)
#只要有一个数据为空，就进行行删除操作
datas=new_df.dropna(axis=0,how='any')
#观察数据的多种统计指标（只能看数值型的 本来9个特征变为7个）
#print(datas.describe().T) #进行转置是为了观察方便

#需求:构建时间和功率之间的映射关系。可以认为：特征属性为时间，目标属性为功率值
#获取x和y变量，并将时间转换为数值型连续变量
def date_format(dt):
    #dt显示是一个Series
    #print(dt.index) #Index(['Date', 'Time'], dtype='object')
    # Date
    # 16 / 12 / 2006
    # Time
    # 17: 24:00
    # Name: 0, dtype: object
    #print(dt)
    t=time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    return t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec
X=datas.iloc[:,0:2]
#print(X)
X=X.apply(lambda x:pd.Series(date_format(x)),axis=1)
print(X)
Y=datas['Global_active_power']

#对数据集进行测试集、训练集的划分
#X:特征矩阵（类型一般为DataFrame）
#Y:特征对应的label标签或者目标属性（类型一般为Series）

# test_size: 对X/Y进行划分的时候，测试集合的数据占比, 是一个(0,1)之间的float类型的值
# random_state: 数据分割是基于随机器进行分割的，该参数给定随机数种子；
# 给一个值(int类型)的作用就是保证每次分割所产生的数数据集是完全相同的
# 默认的随机数种子是当前时间戳 random_state=None的情况下
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,X_test.shape)

#查看训练集上的数据信息
print(X_train.describe().T)

#特征数据标准化（也可以使正常化、归一化、正规化）
# StandardScaler:将数据转换为标准差为1的数据（有一个数据的映射）
# scikit-learn中：如果一个API名字有fit，那么就有模型训练的含义，没法返回值
# scikit-learn中：如果一个API名字中有transform， 那么就表示对数据具有转换的含义操作
# scikit-learn中：如果一个API名字中有predict，那么就表示进行数据预测，会有一个预测结果输出
# scikit-learn中：如果一个API名字中既有fit又有transform的情况下，那就是两者的结合(先做fit，再做transform)

#模型对象创建
ss=StandardScaler()
#训练数据并转化训练集
X_train=ss.fit_transform(X_train)
#直接使用在模型构建数据上进行一个数据标准化操作(测试集)
X_test=ss.transform(X_test)

#<class 'numpy.ndarray'>
print(type(X_train))
#(800, 6) 2
print(X_train.shape,X_train.ndim)
print(pd.DataFrame(X_train).describe().T)

#模型训练
#模型对象构建
# fit_intercept fit训练 intercept截距
# LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# n_jobs模型训练的任务数 是否并行 并行需要至少2个cpu的 基本没什么用这个参数
lr=LinearRegression(fit_intercept=True)
#训练数据
lr.fit(X_train,Y_train)
#预测结果
y_predict=lr.predict(X_test)

#回归中的R^2就是准确率
print('训练集上准确率：',lr.score(X_train,Y_train))
print('测试集上准确率：',lr.score(X_test,Y_test))
#预测值与实际值的差值，平方和 再求平均
mse=np.average((y_predict-Y_test)**2)
rmse=np.sqrt(mse)
print('rmse:',rmse)

#输出模型训练得到的相关参数
print('模型的系数：')
print(lr.coef_)
print('模型的截距：')
print(lr.intercept_)

#模型保存/持久化（跳过）
#加载模型并进行预测（跳过）

#预测值与实际值画图比较
t=np.arange(len(X_test))
#建一个画布 facecolor是背景色
plt.figure(facecolor='w')
plt.plot(t,Y_test,'r-',linewidth=2,label='真实值')
plt.plot(t,y_predict,'g-',linewidth=2,label='预测值')
#显示图例，设置图例的位置
plt.legend(loc='upper lefe')
plt.title('"线性回归预测时间和功率之间的关系',fontsize=20)
#加网格
plt.grid(b=True)
plt.show()


#功率与电流之间的关系
X=datas.iloc[:,2:4]
Y2=datas.iloc[:,5]

#数据分割
X2_train,X2_test,Y2_train,Y2_test=train_test_split(X,Y2,test_size=0.2,random_state=0)

#数据归一化
scaler2=StandardScaler()
X2_train=scaler2.fit_transform(X2_train) #训练并转换
X2_test=scaler2.transform(X2_test) #直接使用在模型构建数据上进行一个数据标准化操作

#模型训练
lr2=LinearRegression()
lr2.fit(X2_train,Y2_train)

#结果预测
Y2_predict=lr2.predict(X2_test)

#模型评估
print('电流预测准确率：',lr2.score(X2_test,Y2_test))
print('电流参数：',lr2.coef_)

#绘制图表
#电流关系
t=np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t,Y2_test,'r-',linewidth=2,label='真实值')
plt.plot(t,Y2_predict,'g-',linewidth=2,label='预测值')
plt.legend(loc='lower right')
plt.title('线性回归预测功率与电流之间的关系',fontsize=20)
plt.grid(b=True)
plt.show()




