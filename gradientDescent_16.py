# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/28 0028 上午 9:27
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 梯度下降算法原理和代码实战
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#防止绘图时中文乱码
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#构建一维的元素图像
def f1(x):
    return 0.5*(x-0.25)**2
def h1(x):
    return 0.5*2*(x-0.25)

#使用梯度下降算法进行解答
GD_X=[]
GD_Y=[]

x=4 #初始值
alpha=0.5
f_change=f1(x)
f_current=f_change
GD_X.append(x)
GD_Y.append(f_current)

#迭代次数
iter_num=0

while f_change>1e-10 and iter_num<100:
    iter_num+=1
    x=x-alpha*h1(x) #迭代参数theata
    tmp=f1(x)
    #判断y值的变化， 不能太小， 太小的话我们认为已经达到最优
    f_change=np.abs(f_current-tmp)
    f_current=tmp
    GD_X.append(x)
    GD_Y.append(f_current)

print(u'最终的结果：(%.5f,%.5f)'%(x,f_current))
print(u'迭代次数是：%d'%iter_num)
print(GD_X)

#构建数据
X=np.arange(-4,4.5,0.05)
Y=np.array(list(map(lambda t:f1(t),X)))
#print(Y)

#画图
plt.figure(facecolor='w')
plt.plot(X,Y,'r-',linewidth=2)
plt.plot(GD_X,GD_Y,'bo--',linewidth=2)
plt.title(u'函数$y=0.5*(x-0.25)^2$ 学习率：%.3f；最终解：(%.3f, %.3f)；迭代次数:%d' % (alpha, x, f_current, iter_num))
plt.show()


#梯度案例实现2
# 在求解机器学习算法的模型参数，即无约束优化问题时，
# 梯度下降（Gradient Descent）是最常采用的方法之一，另一种常用的方法是最小二乘法
from mpl_toolkits.mplot3d import Axes3D

#二维原始图像
def f2(x,y):
    return 0.6*(x+y)**2-x*y
#导函数 偏导
def hx2(x,y):
    return 0.6 * 2 * (x + y) - y
def hy2(x,y):
    return 0.6 * 2 * (x + y) - x

#使用梯度下降算法求解
GD_X1=[]
GD_X2=[]
GD_Y=[]

x1=4
x2=2
alpha=0.5
f_change=f2(x1,x2)
f_current=f_change
GD_X1.append(x1)
GD_X2.append(x2)
GD_Y.append(f_current)

iter_num=0
while f_change>1e-10 and iter_num<100:
    iter_num+=1
    prex1=x1
    prex2=x2
    x1=x1-alpha*hx2(prex1,prex2)
    x2=x2-alpha*hy2(prex1,prex2)

    tmp=f2(x1,x2)
    f_change=np.abs(f_current-tmp)
    f_current=tmp
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_current)
print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x1, x2, f_current))
print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
print(GD_X1)

#构建数据
x1=np.arange(-4,4.5,0.2)
x2=np.arange(-4,4.5,0.2)

x1,x2=np.meshgrid(x1,x2)
Y=np.array(list(map(lambda t:f2(t[0],t[1]),zip(x1.flatten(),x2.flatten()))))
Y.shape=x1.shape

#画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(x1, x2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.plot(GD_X1, GD_X2, GD_Y, 'bo--')

#这行代码出错 还没找见原因 所以注释掉了
#ax.set_title(u'函数$y=0.6 * (θ1 + θ2)^2 - θ1 * θ2$;\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d'% (alpha, x1, x2, f_current, iter_num))
plt.show()

#梯度案例实现3(更加复杂的一个函数)

# 二维原始图像
def f2(x, y):
    return 0.15 * (x + 0.5) ** 2 + 0.25 * (y - 0.25) ** 2 + 0.35 * (1.5 * x - 0.2 * y + 0.35) ** 2


## 偏函数
def hx2(x, y):
    return 0.15 * 2 * (x + 0.5) + 0.25 * 2 * (1.5 * x - 0.2 * y + 0.35) * 1.5


def hy2(x, y):
    return 0.25 * 2 * (y - 0.25) - 0.25 * 2 * (1.5 * x - 0.2 * y + 0.35) * 0.2


# 使用梯度下降法求解
GD_X1 = []
GD_X2 = []
GD_Y = []
x1 = 4
x2 = 4
alpha = 0.5
f_change = f2(x1, x2)
f_current = f_change
GD_X1.append(x1)
GD_X2.append(x2)
GD_Y.append(f_current)
iter_num = 0
while f_change > 1e-10 and iter_num < 100:
    iter_num += 1
    prex1 = x1
    prex2 = x2
    x1 = x1 - alpha * hx2(prex1, prex2)
    x2 = x2 - alpha * hy2(prex1, prex2)

    tmp = f2(x1, x2)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_current)
print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x1, x2, f_current))
print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
print(GD_X1)

# 构建数据
X1 = np.arange(-4, 4.5, 0.2)
X2 = np.arange(-4, 4.5, 0.2)
X1, X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t: f2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
Y.shape = X1.shape

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.plot(GD_X1, GD_X2, GD_Y, 'ko--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_title(u'函数;\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d' % (alpha, x1, x2, f_current, iter_num))
plt.show()