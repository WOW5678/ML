# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/4 0004 上午 8:50
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: softmax回归算法与实例案例：葡萄酒质量分类
 基于葡萄酒数据进行葡萄酒质量预测模型构建，使用SoftMax算法构建模型，
并获取模型构建的效果（类别有7类 k=7）
数据来源：http://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import  ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#设置字符集 防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

##拦截异常
warnings.filterwarnings(action='ignore',category=ConvergenceWarning)

#读取数据
path1='data/winequality-red.csv'
df1=pd.read_csv(path1,sep=';')
df1['type']=1 #设置数据类型为葡萄酒

path2='data/winequality-white.csv'
df2=pd.read_csv(path2,sep=';')
df2['type']=2 #设置数据类型为白酒

#合并两个df
df=pd.concat([df1,df2],axis=0)

#自变量名称
names=["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]
#因变量名称
quanlity='quality'
#显示
print(df.head(5))

#异常数据处理
new_df=df.replace('?',np.nan)
datas=new_df.dropna(how='any') #只要有列为空，就进行删除操作
print('原始数据条数:%d,数据处理之后：%d,异常数据：%d'%(len(df),len(datas),len(df)-len(datas)))

#提取自变量和因变量
x=datas[names]
y=datas[quanlity]

#数据分割
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print('训练数据条数：%d,数据特征数：%d，测试数据条数：%d'%(x_train.shape[0],x_train.shape[1],x_test.shape[0]))

# 2.数据格式化（归一化）
#将数据缩放到[0,1]区间，之前是使用了StandardScaler这个类
ss=MinMaxScaler()
x_train=ss.fit_transform(x_train) #训练数据及归一化数据
x_test=ss.transform(x_test)

#查看y值得范围和数理统计
print(y_train.value_counts())

# 3.模型构建以及训练

# LogisticRegression中，参数说明：
# 1、penalty：惩罚项方式，即使用何种方式进行正则化操作(可选: l1或者l2)
# 2、C ：惩罚项系数，即L1或者L2正则化项中给定的那个λ系数(课件图片上)
# LogisticRegressionCV中，参数说明：
# 1、LogisticRegressionCV表示LogisticRegression进行交叉验证选择超参（机器学习调参）(惩罚项系数C或者λ)
# 2、Cs ：表示在交叉验证中惩罚项系数λ的可选范围

## multi_class: 分类方式参数；参数可选: ovr(默认)、multinomial；这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
### ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类 里面会有多个模型
### multinomial: many-vs-many（MVM）,即Softmax分类效果 只有一个模型
## class_weight: 特征权重参数

### Softmax算法相对于Logistic算法来讲，在sklearn中体现的代码形式来讲，主要只是参数的不同
# Softmax k个θ向量并不是表示有k个模型 底层只有一个模型 在模型中训练的是θ矩阵而非向量
lr=LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5,1,100),
                        multi_class='multinomial',penalty='l2',solver='lbfgs')
lr.fit(x_train,y_train)

# 4.模型效果获取
r=lr.score(x_train,y_train)
print('R值：',r)
print("特征稀疏化比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print("参数：",lr.coef_)
print("截距：",lr.intercept_)

print("概率：", lr.predict_proba(x_test))
# 获取概率函数返回的概率值
print("概率有多少：", lr.predict_proba(x_test).shape)

# 数据预测
## a. 预测数据格式化(归一化)
## b. 结果数据预测
y_predict = lr.predict(x_test)


## c. 图表展示
x_len = range(len(x_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(-1,11)
plt.plot(x_len, y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, y_predict, 'go', markersize = 12, zorder=2, label=u'预测值准确率,$R^2$=%.3f' % lr.score(x_train, y_train))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号ID', fontsize=18)
plt.ylabel(u'葡萄酒质量层级', fontsize=18)
plt.title(u'葡萄酒质量预测分析', fontsize=20)
plt.show()
