import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

links = [
  './model1539594715.ckpt',
  './model1539604193.ckpt',
  './model1539616437.ckpt',
  './model1539620015.ckpt',
  './model1539623326.ckpt',
  './model1539628351.ckpt',
  './model1539632040.ckpt',
  './model1539635508.ckpt',
  './model1539640535.ckpt',
  './model1539644068.ckpt',
  './model1539648107.ckpt',
  './model1539651922.ckpt',
  './model1539655746.ckpt',
  './model1539659236.ckpt',
  './model1539662734.ckpt',
  './model1539668023.ckpt',
  './model1539672415.ckpt',
  './model1539675574.ckpt',
  './model1539680202.ckpt',
  './model1539684872.ckpt',
  './model1539688714.ckpt'
]

M_cnt = len(links)

def load(link):
  with tf.Session() as new_sess:
    saver = tf.train.Saver()
    saver.restore(new_sess,link)
    a,b,c = run_model(new_sess,y_out_multiple,mean_loss,X_test,y_test,1,64)
    return c

ans = []
predict = []

for i in range(M_cnt):
  ans.append(load(links[i]))

ans = np.array(ans,dtype='int64')

for i in range(ans.shape[1]):
  predict.append(np.bincount(ans[:,i]).argmax())

predict = np.array(predict)
accuracy = float(np.sum(predict==y_test))/float(ans.shape[1])
print('Accuracy %d'%accuracy)