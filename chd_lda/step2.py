# -*- coding:utf-8 -*-  
'''
Created on 2020年9月15日

@author: zjf

手动实现LDA模型
'''

import numpy as np  

                
class LDAModel(object):  

    def __init__(self, num_topics, beta, alpha, iter_times, docs):  
        '''
                    模型参数  
                    聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)  
        '''
        self.docs = docs
        self.words_count = len(word2id)  # 词汇数  
        self.docs_count = len(docs)  # 文档数  
       
        self.K = num_topics 
        self.beta = beta 
        self.alpha = alpha 
        self.iter_times = iter_times  
        
        # nw,词word在主题topic上的分布  
        self.nw = np.zeros((self.words_count, self.K), dtype="int")  
        # nd,每个doc中各个topic的词的总数  
        self.nd = np.zeros((self.docs_count, self.K), dtype="int")  
        # M*doc.size()，文档中词的主题分布  
        self.Z = np.array([ [0 for y in range(len(self.docs[x]))] for x in range(self.docs_count)])  
  
        # 随机先分配类型，为每个文档中的各个单词分配主题  
        for x in range(len(self.Z)):  
            for y in range(len(self.docs[x])):  
                topic = np.random.randint(0, self.K - 1)  # 随机取一个主题  
                self.Z[x][y] = topic  # 文档中词的主题分布  
                self.nw[self.docs[x][y]][topic] += 1  
                self.nd[x][topic] += 1  
        
        for x in range(self.iter_times):  # 循环多次采样
            for i in range(self.docs_count):  
                for j in range(len(self.docs[i])):  
                    topic = self.sampling(i, j)  
                    self.Z[i][j] = topic  
        
        # 计算每个主题的词分布
        self.phi = np.array([ [ 0.0 for y in range(self.words_count) ] for x in range(self.K)]) 
        _nwsum = np.sum(self.nw, axis=0)
        for i in range(self.K):  
            self.phi[i] = (self.nw.T[i] + self.beta) / (_nwsum[i] + self.words_count * self.beta)  
            
        # 计算每个文档的主题分布
        self.theta = np.array([ [0.0 for y in range(self.K)] for x in range(self.docs_count) ])  
        _ndsum = np.sum(self.nd, axis=1)
        for m in range(self.docs_count):
            for k in range(self.K):
                self.theta[m][k] = (self.nd[m][k] + self.alpha) / (_ndsum[m] + self.K * self.alpha)
          
    def sampling(self, i, j):  # Gibbs Sampling
        topic = self.Z[i][j]  
        word = self.docs[i][j]  
        self.nw[word][topic] -= 1  
        self.nd[i][topic] -= 1  
        
        _nwsum = np.sum(self.nw, axis=0)
        _ndsum = np.sum(self.nd, axis=1)
  
        Vbeta = self.words_count * self.beta  
        Kalpha = self.K * self.alpha  
        p = (self.nd[i] + self.alpha) / (_ndsum[i] + Kalpha) * (self.nw[word] + self.beta) / (_nwsum + Vbeta) 
  
        # 按这个更新主题更好理解，这个效果还不错  
        p = np.squeeze(np.asarray(p / np.sum(p)))  
        topic = np.argmax(np.random.multinomial(1, p))  
  
        self.nw[word][topic] += 1  
        self.nd[i][topic] += 1  
        return topic  
    
    def get_topic_terms(self, topicid):
        twords = [(n, self.phi[topicid][n]) for n in range(len(self.phi[topicid]))]  
        twords.sort(key=lambda i:i[1], reverse=True)  
        return twords
    
    def get_document_topics(self):
        lst = []
        for x in range(len(docs)):  
            topic = set()
            for y in range(len(docs[x])):  
                topic.add(lda.Z[x][y])
            lst.append(topic)
        return lst


def load_stopword():
    f_stop = open('stopword.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


if __name__ == '__main__':  
    np.random.seed(0)
    
    # 载入文件，并读入数组中
    stop_words = load_stopword()
    f = open('LDA_test.txt', encoding='utf-8')  # LDA_test.txt
    texts = [[word for word in line.strip().lower().split() if word not in stop_words] for line in f]
    f.close()
    
    # exact all words from text
    word2id = {}
    items_idx = 0  
    for line in texts:  
        for item in line:  
            if item not in word2id:  # 已有的话，只是当前文档追加  
                word2id[item] = items_idx  
                items_idx += 1  
                
    # 将文本中的词全部根据word2id替换成编号
    docs = [[word2id[item] for item in line] for line in texts]  
    
    num_topics = 5
    lda = LDAModel(num_topics, 0.01, 0.01, 100, docs)
    
    print('每个主题的词分布：')
    worddic = {value:key for key, value in word2id.items()}
    for topic_id in range(num_topics):  
        terms = lda.get_topic_terms(topicid=topic_id)
        
        lst = [worddic[t[0]] for t in terms[:5]]
        print('主题#%d：\t' % topic_id, " ".join(lst))
        
    print('文档的主题分布：')
    doc_topics = lda.get_document_topics()
    for x in range(len(docs)):  
        topic = doc_topics[x]
        print("第%d个文档的前10个主题：" % x, list(topic))
