# -*- coding:utf-8 -*-
'''
Created on 2017年6月12日

@author: Jeffrey Zhou
'''
import re
from numpy import * 
    
def loadDataSet1():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 1 is abusive ,0 not
    return postingList, classVec

def loadDataSet2():
    postingList = [];
    classVec = [];
    for i in range(1, 26):
        filetoken = loadFileToken('email/ham/%d.txt' % i)
        postingList.append(filetoken)
        classVec.append(0)
    for i in range(1, 26):
        filetoken = loadFileToken('email/spam/%d.txt' % i)
        postingList.append(filetoken)
        classVec.append(1)
    return postingList, classVec

def loadFileToken(filename):
    str = open(filename).read()
    listofTokens = re.split(r'\W', str) # 用正则表达式拆分单词
    listofTokens2 = [tok.lower() for tok in listofTokens if len(tok) > 2] # 遍历单词，排除掉长度小于3的词
    return listofTokens2

def createVocabList(dataSet):
    vacabSet = set([]) # 相当于hash
    for document in dataSet:
        vacabSet = vacabSet | set(document) # 合并两个集合
    return list(vacabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print( "the word :%s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(tranMatrix, trainCategory):
    numTrainDocs = len(tranMatrix)
    numWords = len(tranMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords) # 4行，初使化p(wi|c1) 和p(wi|c0),为了防止多个概率相乘中间有0，分子分母都加1
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs): # 循环训练集
        if trainCategory[i] == 1:
            p1Num += tranMatrix[i]
            p1Denom += sum(tranMatrix[i])
        else:
            p0Num += tranMatrix[i]
            p0Denom += sum(tranMatrix[i])
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    return p0Vec, p1Vec, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * log(p1Vec)) + log(pClass1)
    p0 = sum(vec2Classify * log(p0Vec)) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
if __name__ == '__main__':
    #test
    tonen = loadFileToken('email/ham/7.txt')
    print (tonen)
    #test
    listOposts, listClasses = loadDataSet2();
    print(listOposts)
    print(listClasses)
    #test
    myVocabList = createVocabList(listOposts)
    print(myVocabList)
    #test
    returnVec = setOfWords2Vec(myVocabList, listOposts[0]) #测试下第一个项量情况
    print(returnVec)
    
    # test
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    print('pAb=', pAb)
    print('p0v=', p0v)
    print('p1v=', p1v)

    #test
    testEntry = loadFileToken('email/ham/7.txt')
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    ret = classifyNB(thisDoc, p0v, p1v, pAb)
    print('result1 ', ret)
     
    testEntry = loadFileToken('email/spam/7.txt')
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    ret = classifyNB(thisDoc, p0v, p1v, pAb)
    print('result2 ', ret)