# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/20 11:16
'''

import pandas as pd
from datetime import datetime

# 业务分析:五年数据集，按小时来报告天气和污染水平，

# 解决问题：根据过去几个小时的天气条件和污染状况预测下一个小时的污染情况，（当然，也可以是下几个小时的污染情况）

def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')

# 导入数据集，
dataset=pd.read_csv('raw.csv',parse_dates=[['year','month','day','hour']],index_col=0,infer_datetime_format=True)
dataset.drop('No',axis=1,inplace=True)  # 去除No列

# 列此时只有，Columns: [pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir]
# 修改一下列名，让列名可读性更强
dataset.columns=['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
dataset.index.names=['date']   # 修改索引的列名
dataset['pollution'].fillna(0,inplace=True)  # 对pollution列的空值用0填充
# 去除前24小时的数据
dataset=dataset[24:]
dataset.to_csv('pollution.csv')
print(dataset.head())

# 查看前几行
#             pollution  dew  temp   press wnd_dir  wnd_spd  snow  rain
# date
# 2010 1 2 0      129.0  -16  -4.0  1020.0      SE     1.79     0     0
# 2010 1 2 1      148.0  -15  -4.0  1020.0      SE     2.68     0     0
# 2010 1 2 2      159.0  -11  -5.0  1021.0      SE     3.57     0     0
# 2010 1 2 3      181.0   -7  -5.0  1022.0      SE     5.36     1     0
# 2010 1 2 4      138.0   -7  -5.0  1022.0      SE     6.25     2     0





































