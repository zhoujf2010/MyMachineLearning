# coding=utf-8
'''
Created on 2018年3月29日

@author: zjf

TF-IDF Demo
'''
import numpy as np
from time import time
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

if __name__ == '__main__':
    print(u'开始下载/加载数据...')
    t_start = time()
    # remove = ('headers', 'footers', 'quotes')
    remove = ()
    categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'
    # categories = None     # 若分类所有类别，请注意内存是否够用
    data_train = datasets.fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0, remove=remove)
    t_end = time()
    print(u'下载/加载数据完成，耗时%.3f秒' % (t_end - t_start))
    print(u'数据类型：', type(data_train))
    print(u'训练集包含的文本数目：', len(data_train.data))
    print(u' -- 前10个文本 -- ')
    categories = data_train.target_names
    pprint(categories)
    y_train = data_train.target
    for i in np.arange(10):
        print( u'文本%d(属于类别 - %s)：' % (i+1, categories[y_train[i]]))
        print(data_train.data[i])
        print('\n\n')
    vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.5, sublinear_tf=True)
    x_train = vectorizer.fit_transform(data_train.data)  # x_train是稀疏的，scipy.sparse.csr.csr_matrix
    print(u'训练集样本个数：%d，特征个数：%d' % x_train.shape)
    print(u'停止词:\n',)
    pprint(vectorizer.get_stop_words())   
#     vectorizer.transform('') #新词的转换   
        
    
    