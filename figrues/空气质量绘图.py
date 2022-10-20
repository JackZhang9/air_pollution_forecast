# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/20 16:33
'''

import pandas as pd
from matplotlib import pyplot as plt


# 导入数据
dataset=pd.read_csv('pollution.csv',header=0,index_col=0)
values=dataset.values  # 获得除索引和列名之外的数据,是ndarray
# 画图
colors=['r','g','b','y','c','m','k','r']
plt.figure(figsize=(35,15),dpi=80)
i=1
for idx in range(8):
   plt.subplot(8,1,i)
   plt.plot(values[:,idx],c=colors[idx])  # 每一列的所有行
   plt.title(dataset.columns[idx],y=0.5,loc='right')
   i+=1
plt.savefig('空气污染时间序列绘图.png')
plt.show()

















