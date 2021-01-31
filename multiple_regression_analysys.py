# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:59:24 2021

@author: marlm
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#データの読み込み
from sklearn.datasets import load_boston
boston = load_boston()
boston

boston_df_ob = pd.DataFrame(boston.target,columns=["PRICE"])
boston_df_ob

boston_df_ex = pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df_ex

boston_df = pd.concat([boston_df_ob,boston_df_ex],axis=1)
boston_df

#Numpyの多次元配列arrayに変換
boston_arr = np.array(boston_df)
print(boston_arr)

#分散共分散行例の算出
cov = np.cov(boston_arr,rowvar=0,bias=0)
print(cov)

#分散共分散行列Aの作成
covA = cov[1:,1:]
#分散共分散行列Bの作成
covB = cov[1:,0] #1×13
covB = covB.reshape(13,1) #13×1の2次元配列に変換

#逆行列の作成
covA_inv = np.linalg.inv(covA)

#偏回帰係数の計算
coef = np.dot(covA_inv,covB)
print(coef)

#切片の導出
#説明変数それぞれの平均値
x_mean = np.mean(boston_arr[:,1:],axis=0)
#回帰係数×説明変数の総和
sum = 0
for i,j in zip(coef,x_mean):
  sum += i*j
#目的変数の平均値から説明変数の総和をひく
intersept = np.mean(boston_arr[:,0]) - sum

##予測値の分散を求める
#説明変数xのみの行列取得
x = boston_arr[:,1:]
#変数毎に偏回帰係数を乗算し、切片を加算
coef2 = np.reshape(coef,[1,13])
y_hat = np.sum(x*coef2,axis=1)+intersept

np.var(y_hat)

y = boston_arr[:,0]
np.var(y)

R = np.var(y_hat)/np.var(y)
print("決定係数R:",R)

