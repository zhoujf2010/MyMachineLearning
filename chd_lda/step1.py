'''
Created on 2020年9月15日

@author: zjf
'''

import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time
from random import random


def load_stopword():
    f_stop = open('stopword.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


if __name__ == '__main__':
    print('初始化停止词列表 --')
    t_start = time.time()
    stop_words = load_stopword()

    print('开始读入语料数据 -- ')
    f = open('LDA_test.txt', encoding='utf-8')  # LDA_test.txt
    texts = [[word for word in line.strip().lower().split() if word not in stop_words] for line in f]
    f.close()
    print('文本数目：%d个' % len(texts))

    print('正在建立词典 --')
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print(u'词的个数：', V)
    print('正在计算文本向量 --')
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('正在计算文档TF-IDF --')
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))
    
    print('LDA模型拟合推断 --')
    num_topics = 10
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                            alpha=0.01, eta=0.01, minimum_probability=0.001,
                            update_every=1, chunksize=100, passes=1)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))
    
    print('每个主题的词分布：')
    for topic_id in range(num_topics):
        terms = lda.get_topic_terms(topicid=topic_id)
        
        lst = [dictionary.id2token[t[0]] for t in terms[:7]]  # 取出前7个主题词
        print('主题#%d：\t' % topic_id," ".join(lst))
        
    # 随机打印某10个文档的主题
    num_show_topic = 10  # 每个文档显示前几个主题
    print('10个文档的主题分布：')
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    
    for _ in range(10):
        index = np.random.randint(0, len(texts))
        topic = np.array(doc_topics[index])
        topic_distribute = np.array(topic[:, 1])
        topic_idx = topic_distribute.argsort()[:-11:-1]
        print(('第%d个文档的前10个主题：' % (index)), topic_idx)
    
#     similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
#     print( 'Similarity:')
#     pprint(list(similarity))
