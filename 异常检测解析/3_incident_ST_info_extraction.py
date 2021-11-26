# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:59:30 2020

@author: senda
"""
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pandas as pd
import xgboost


################################################ Incident classificatoin using Weibo data ########################################################

# Read data, two columns: text and label
data = pd.read_excel(r'C:\Users\g\Desktop\anomaly detection\weibo_data\weibo_accident.xls') 
# divided into training set and verification set
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data["微博内容"],data['flag'])
# encoding the label
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

##------------------------------------------------ feature engineering (based TF-IDF） -------------------------
# 方法1：词语级tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data["微博内容"])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# 方法2：ngram级tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(data["微博内容"])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

# 方法3：词性+ngram级tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(data["微博内容"])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

##--------------------------------------------------- classification model ----------------------------------

##---------------示例start-------------
model = xgboost.XGBClassifier(learning_rate=0.1, # 学习率，控制每次迭代更新权重时的步长，默认0.3
                      n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重，值越大，越容易欠拟合；值越小，越容易过拟合
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                      colsample_btree=0.8,       # 随机选择80%特征建立决策树
                      objective='multi:softmax', # 指定目标(损失)函数：多分类
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27            # 随机数
                      )
model.fit(x_train,y_train,
          eval_set = [(x_test,y_test)],
          eval_metric = "error", # 默认二分类
          early_stopping_rounds = 10, #在验证集上，当连续n次迭代，分数没有提高后，提前终止训练
          verbose = True)
##---------------示例end-------------

# 特征为词语级别TF-IDF向量的Xgboost
xgm = xgboost.XGBClassifier()
xgm.fit(xtrain_tfidf.tocsc(), train_y)
y_pred = xgm.predict(xvalid_tfidf.tocsc())
C=confusion_matrix(valid_y, y_pred)  # confusion matrix
print(C, end='\n\n')

# 特征为ngram级别TF-IDF向量的Xgboost
xgm = xgboost.XGBClassifier()
xgm.fit(xtrain_tfidf_ngram.tocsc(), train_y)
y_pred = xgm.predict(xvalid_tfidf_ngram.tocsc())
C=confusion_matrix(valid_y, y_pred)  # confusion matrix
print(C, end='\n\n')

# 特征为词性+ngram级别TF-IDF向量的Xgboost （实验效果最好）
xgm = xgboost.XGBClassifier(learning_rate=0.1, # 学习率，控制每次迭代更新权重时的步长，默认0.3
                      n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重，值越大，越容易欠拟合；值越小，越容易过拟合
                      gamma=0.1,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.9,             # 随机选择80%样本建立决策树
                      colsample_btree=1,       # 选择全部特征建立决策树
                      #objective='multi:softmax', # 指定目标(损失)函数：多分类
                      scale_pos_weight=2)
xgm.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y, 
        eval_set = [(xvalid_tfidf_ngram_chars.tocsc(),valid_y)],
        early_stopping_rounds = 30)
y_pred = xgm.predict(xvalid_tfidf_ngram_chars.tocsc())
C=confusion_matrix(valid_y, y_pred)  # confusion matrix
print(C, end='\n\n')

import matplotlib.pyplot as plt
from xgboost import plot_importance
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(xgm,height=0.5, ax=ax, max_num_features=64)
plt.show()

errors = [0.06452,0.06452,0.06452,0.06452,0.06452,
          0.06452,0.05376,0.06452,0.04301,0.05376,
          0.04301,0.03226,0.03226,0.03226,0.04301,
          0.04301,0.03226,0.04301,0.04301,0.03226,
          0.03226,0.03226,0.03226,0.03226,0.03226,
          0.03226,0.03226,0.03226,0.03226,0.03226,
          0.03226,0.03226,0.03226,0.03226,0.03226,
          0.03226,0.03226,0.03226,0.03226,0.03226,
          0.02150,0.02150,0.02150,0.02150,0.02150,
          0.02150,0.02150,0.02150,0.02150,0.02150,
          0.02150,0.02150,0.02150,0.02150,0.02150,
          0.02150,0.02150,0.02150,0.02150,0.02150,
          0.02150,0.02150,0.02150,0.02150,0.02150,
          0.02150,0.02150,0.02150,0.02150,0.02150]
fig,ax = plt.subplots(figsize=(7,5))
plt.plot(errors)
plt.axvline(40,color='r')
plt.xlabel('Round',fontsize=14)
plt.ylabel('Gains',fontsize=14)
plt.savefig(r'C:\Users\g\Desktop\anomaly detection\pic\classifier_valid',dpi = 300,bbox_inches='tight')
plt.show()
################################################## Information extraction for incident-related Weibo ############################################

## ----------------------------------------------- Named entity recognition (NER) -----------------------------------
import jieba
from jieba import posseg as pseg

# 抽取关键词的类初始化时默认加载了分词函数（分词是抽取关键词的基础）、词性标注函数和停用词表
# 若用默认文件不能满足需求，可以选择性自定义IDF频率文件，停用词文件
#analyse.set_idf_path("../extra_dict/idf.txt.big")
#analyse.set_stop_words("../extra_dict/stop_words.txt") #此函数执行后，会更新默认的停用词集合

# 选择性加载自定义语库（UTF-8编码格式），根据个人需求提高分词准确率，加载是对默认词库的扩展 
# 或者对C:\Users\g\Anaconda3\Lib\site-packages\jieba\dict.txt进行更新
jieba.load_userdict(r'C:\Users\g\Desktop\anomaly detection\python_code\tdata\dict_links.txt')

                       
text = "志新路发生了严重的交通事故"

# 精准模式cut_all=False
str_jing1=jieba.cut(text,cut_all=False)
print('精准模式分词：{ %d}' % len(list(str_jing1)))
str_jing2=jieba.cut(text,cut_all=False)
print("$".join(str_jing2))

# 根据词性选择表示位置的名词
locations = []
strings = [[s.word,s.flag] for s in pseg.cut(text)] # 基于精准模式分词的结果
for s in strings:
    if(s[1]=='nrs'):
        locations.append(s[0])
        print(s)

## ----------------------------------------------- Locations geocoding -----------------------------------
import requests
def geocode(address):
    parameters = {'address': address, 'key': 'cc549ef9ce7352a8d668b28b15468741'}
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    response = requests.get(url, parameters)
    # answer = response.geocodes['location']
    # print(answer)
    answer = response.json()
    if answer['count'] == str(1):
        location = answer['geocodes'][0]['location'].split(',')
        lng = float(location[0])       #经度
        lat = float(location[1])       #纬度
    return [lng,lat]

for loc in locations:
    print(loc,geocode(loc))











