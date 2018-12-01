# -*- coding:utf-8 -*-
'''
Create time: 2018/12/1 10:03
@Author: 大丫头
BGD,普通最小二乘法求线性回归问题的实现与对比
'''
# 基于梯度下降算法编写实现线性回归，并自行使用模拟数据进行测试，
# 同时对同样的模拟数据进行两种算法的比较（scikit-learn中的普通最小二乘LinearRegression 与 自己实现的梯度下降算法）

'''实现一个批量梯度下降算法秋节线性回归问题模型'''
import math
import numpy as np

def validate(X,Y):
    '''校验X和Y的格式是否正确'''
    if len(X)!=len(Y):
        raise Exception('样本参数异常')
    else:
        n=len(X[0])
        for l in X:
            if len(l)!=n:
                raise Exception('参数异常')
        if len(Y[0])!=1:
            raise Exception('参数异常')
def predict(x,theata,intercept):
    '''对单个的样本计算出预测值'''
    result=0.0
    #x与theata相乘
    n=len(x)
    for i in range(n):
        result+=x[i]*theata[i]
    #加上截距
    result+=intercept
    return result

def predic_X(X,theata,intercept=0):
    '''根据样本特征和参数预测y值'''
    Y=[]
    for x in X:
        Y.append(predict(x,theata,intercept))
    return Y

def fit(X,Y,alpha=0.01,max_iter=100,fit_intercept=True,tol=1e-4):
    '''
    进行模型的训练，返回模型theata参数值何截距
    :param X:特征属性矩阵，二维矩阵 m*n, m为样本个数，n为特征个数
    :param Y:目标属性矩阵，二维矩阵，m*k,m为样本个数，k表示y值得个数，一般k=1
    :param alpha:学习率 步长 默认0.01
    :param max_iter: 学习率 步长 默认0.01
    :param fit_intercept:是否训练截距 默认True 训练
    :param tol:是否训练截距 默认True 训练
    :return:（theta，intercept）
    '''
    #1、校验一下收入的X、Y数据格式
    validate(X,Y)

    #2、开始训练模型 迭代 计算参数
    #获取行和列 分别记作样本的个数m和特征g个数n
    m,n=np.shape(X)
    #定义需要训练的参数 给定初始值
    theata=[0 for i in range(n)]
    intercept=0

    #定义一个临时的变量
    diff=[0 for i in range(m)]
    max_iter=100 if max_iter<=0 else max_iter

    #开始迭代 更新参数
    for i in range(max_iter):
        #在当前theata的情况下，预测值与真实值之间的差值
        for k in range(m):
            y_true=Y[k][0]
            y_predict=predict(X[k],theata,intercept)
            diff[k]=y_true-y_predict
        #对theata进行更新
        for j in range(n):
            #计算梯度值
            gd=0
            for k in range(m):
                gd+=diff[k]*X[k][j]
            theata[j]+=alpha*gd
        #训练截距的话（相当于求解theata的时候，对应的维度上的x的取值是1）
        if fit_intercept:
            gd=np.sum(diff)
            intercept+=alpha*gd

        #需要判断损失函数的值是否已经收敛（是否小于给定的值）
        #计算损失函数的值
        #判断是否收敛
        sum_j=0.0
        for k in range(m):
            y_true=Y[k][0]
            y_predict=predict(X[k],theata,intercept)
            j=y_true-y_predict
            sum_j+=math.pow(j,2)
        sum_j/=m

        if sum_j<tol:
            break
    #3、返回参数
    return (theata,intercept)
def score(Y,Y_predict):
    '''计算回归模型的R^2值'''
    #1、计算rss与tss
    average_Y=np.average(Y)
    m=len(Y)
    rss,tss=0.0,0.0
    for k in range(m):
        rss+=math.pow(Y[k]-Y_predict[k],2)
        tss+=math.pow(Y[k]-average_Y,2)
    #计算r^2的值
    r_2=1.0-1.0*rss/tss
    return r_2

if __name__ == '__main__':
    #测试一下BGD与sklit-learn里最小二乘法的比较
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    #设置字符集 防止中文乱码
    mpl.rcParams['font.sans-serif']=[u'simHei']
    mpl.rcParams['axes.unicode_minus']=False

    #创建模拟数据（样本数据）
    np.random.seed(0)
    np.set_printoptions(linewidth=1000,suppress=True)

    N=10
    x=np.linspace(0,6,N)+np.random.randn(N)
    y=1.8* x **3 + x**2 - 14*x - 7 + np.random.randn(N)
    x.shape=-1,1
    y.shape=-1,1
    print(x)
    print(y)

    #在样本基础上 进行模型的训练
    lr=LinearRegression(fit_intercept=True)
    lr.fit(x,y)
    print('sklearn 模块自带的最小二乘法实现----')
    s1=score(y,lr.predict(x))
    print('我们自己写的R^2值的计算：%.3f'%s1)
    print('Sklearn自带的R^2值的计算：%.3f'%lr.score(x,y))
    print('参数列表theata:',lr.coef_)
    print('截距：',lr.intercept_)

    #自己写的BGD训练
    theata,intecept=fit(x,y,alpha=0.01,max_iter=100,fit_intercept=True)
    print('自己写的BGD梯度下降算法实现----')
    s2=score(y,predic_X(x,theata,intecept))
    print('我们自己写的R^2值的计算：%.3f'%s2)
    print('参数列表theata:',lr.coef_)
    print('截距：',lr.intercept_)

    #为了直观的比较  开始画图
    plt.figure(figsize=(2,6),facecolor='w')

    #为了画那条直线 需要产生很多的模拟数据
    x_hat=np.linspace(x.min(),x.max(),num=100)
    x_hat.shape=-1,1
    #框架里的最小二乘法
    y_hat=lr.predict(x_hat)
    #自己写的BGD梯度下降
    y_hat2=predic_X(x_hat,theata,intecept)

    plt.plot(x,y,'ro',ms=10,zorder=3)
    plt.plot(x_hat, y_hat, color='g', lw=2, alpha=0.75, label=u'普通最小二乘，准确率$R^2$:%.3f' % s1, zorder=2)
    plt.plot(x_hat, y_hat2, color='b', lw=2, alpha=0.75, label=u'BGD梯度下降，准确率$R^2$:%.3f' % s2, zorder=1)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.suptitle(u'普通最小二乘与BGD梯度下降的线性回归模型比较', fontsize=22)

    plt.show()




