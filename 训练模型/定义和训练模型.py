# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/21 17:47
'''
import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
pd.set_option('display.max_rows', None, 'display.max_columns', None,'display.width',None)

def load_data(path):
    # 加载数据集
    dataset=pd.read_csv(path,header=0,index_col=0)
    values=dataset.values
    return values


def data_split(values):
    # 使用前一年的数据来拟合
    n_train_hours=365*24
    train=values[:n_train_hours,:]
    test=values[n_train_hours:,:]
    # print(train)
    # 划分输入输出
    train_x,train_y=train[:,:-1],train[:,-1]
    test_x,test_y=test[:,:-1],test[:,-1]
    # print(train_x.shape)
    # (8760, 8)  8760个训练数据。每个数据有8个特征
    # 将二维的输入矩阵reshape成三维的张量，符合lstm的输入数据格式，[样本数，时间步长，特征个数]
    train_x=train_x.reshape(train_x.shape[0],1,train_x.shape[1])  # 这里样本数为全部样本，步长为1，特征数为8
    test_x=test_x.reshape(test_x.shape[0],1,test_x.shape[1])
    # print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    # (8760, 1, 8) (8760,) (35039, 1, 8) (35039,)
    return train_x,test_x,train_y,test_y


def model_build(train_x):
    # 模型搭建,训练完后，这几行注释
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    # # print(model.summary())  # 参数32051
    return model


def model_fit(model,train_x,test_x,train_y,test_y):
    # 模型训练,训练完后，这行注释
    history=model.fit(train_x,train_y,batch_size=32,epochs=50,validation_data=(test_x,test_y),verbose=1)
    return history,model


def model_save(model):
    #  模型导出保存，模型保存成功后，使用时就从保存的模型中读取模型，不再重新训练，导出完后，这行注释
    # model.save()保存完整keras模型，模型比tf保存的稍大，保存为pb格式
    model.save(r'myModel/v3/model.h5')
    return None


def model_load():
    # # 加载模型
    model=keras.models.load_model(r'myModel/v3/model.h5')
    return model


def model_predict(model,test_x):
    # print(list(model.signatures.keys()))  # ['serving_default']
    # f=model.signatures['serving_default']
    # pre=f(digits = tf.constant(test_x.tolist()))  # 模型预测
    # print(pre)

    # 成功版本1，先注释，测试第2版本能不能用
    y_pre=model.predict(test_x)
    y_pre=np.reshape(y_pre,(y_pre.shape[0],y_pre.shape[2]))
    # 预测值保存为csv文件
    np.savetxt('pre.csv',y_pre,fmt='%f',delimiter=None)
    print('预测值 已保存为csv文件！')
    print(y_pre,type(y_pre),y_pre.shape)




if __name__ == '__main__':

    path='reframed.csv'
    values=load_data(path)
    train_x,test_x,train_y,test_y=data_split(values)
    # model=model_build(train_x)
    # history,model=model_fit(model,train_x,test_x,train_y,test_y)
    # model_save(model)  # 模型已保存，不需要再次保存

    # 直接调用训练好的pd格式模型
    model2=model_load()
    model_predict(model2,test_x)






