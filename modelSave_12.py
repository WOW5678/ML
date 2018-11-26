# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/19 0019 上午 9:11
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 最小二乘公式法和模型部署：持久化与加载使用

"""
from sklearn.model_selection import train_test_split
#数据标准化
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
from pandas import DataFrame

#设置字体 防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#加载数据
## 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
path1='data/household_power_consumption_1000.txt'
df=pd.read_csv(path1,sep=';',low_memory=False)
print(df.head(2))

#功率和电流之间的关系
x2=df.iloc[:,2:4]
print(x2.shape)
Y2=df.iloc[:,5]

#数据分割
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,Y2,test_size=0.2,random_state=0)

#模型对象创建
ss=StandardScaler()
#训练模型并转换训练集
x2_train=ss.fit_transform(x2_train)
#直接使用在模型数据上进行一个数据标准化操作（测试集）
x2_test=ss.transform(x2_test)
print(type(x2_test))

#将x和y转换成矩阵的形式
X=np.mat(x2_train)
Y=np.mat(y2_train).reshape(-1,1)
print(type(X))

#使用公式计算theat
# .I表示矩阵的逆
theta=(X.T*X).I*X.T*Y
print(theta)

#对测试集合进行测试
y_hat=np.mat(x2_test)*theta

#画图看看
#电流关系
t=np.arange(len(x2_test))
plt.figure(facecolor='w')
plt.plot(t,y2_test,'r-',linewidth=2,label=u'真实值')
plt.plot(t,y_hat,'g-',linewidth=2,label=u'预测值')
plt.legend(loc='lower right')
plt.title(u'线性回归预测功率与电流之间的关系',fontsize=20)
plt.grid(b=True)
plt.show()

#模型保存/持久化
# 在机器学习部署的时候，实际上其中一种方式就是将模型进行输出；另外一种方式就是直接将预测结果输出数据库
# 模型输出一般是将模型输出到磁盘文件
from sklearn.externals import joblib

#保存模型要求给定的文件所在的文件夹必须存在
joblib.dump(ss,'result/data_ss.model') #将标准化模型保存
#joblib.dump(lr,'result/data_lr.model') #将模型墨村

#加载模型
ss3=joblib.load('result/data_ss.model') #加载模型
#lr3=joblib.load('result/data_lr.model') #加载模型
#使用加载的模型进行预测
data2 = [[12, 17]]
data2 = ss3.transform(data2)
print(data2)
# print(lr3.predict(data2))