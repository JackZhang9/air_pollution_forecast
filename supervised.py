# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/24 10:43
'''

from pandas import concat,DataFrame

# 自定义的专门用来处理各种数据，多变量，单变量的一个工具类

# 多变量数据
# 使用series_to_supervised()函数转换数据集，通过这个函数可以将，数据变得t-1和t
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



# 单变量预测，单变量的一个延迟步长t-1来预测当前时间步长t
def series_to_supervised_one(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	example1:
	    one step length:
	       var1(t-1)  var1(t)
        1        0.0        1
        2        1.0        2
        3        2.0        3
        4        3.0        4
        5        4.0        5
        6        5.0        6
        7        6.0        7
        8        7.0        8
        9        8.0        9
    example2:
        three steps length:
	        var1(t-3)  var1(t-2)  var1(t-1)  var1(t)
        3        0.0        1.0        2.0        3
        4        1.0        2.0        3.0        4
        5        2.0        3.0        4.0        5
        6        3.0        4.0        5.0        6
        7        4.0        5.0        6.0        7
        8        5.0        6.0        7.0        8
        9        6.0        7.0        8.0        9
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg





