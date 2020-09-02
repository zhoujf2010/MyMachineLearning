'''
Created on 2020年8月27日

@author: zjf

自行实现apriori代码
'''
import numpy as np
import  pandas as pd
from itertools import combinations

def apriori(df, minSupport=0.5):
    # 得到反过来的map，id-item
    mapping = {idx: item for idx, item in enumerate(df.columns)}
    
    # 初使列表（即所有商品的集合）
    checkset = np.eye(len(df.columns)).tolist()
    data = []
    
    while len(checkset) > 0:
        # 排除商品集合（checkset）中最小支持度的项
        retset = np.zeros(len(checkset))
        for idx, item in enumerate(checkset):
            for row in df.values:
                if np.sum(np.logical_and(item, row)) > np.sum(item) - 1:
                    retset[idx] += 1
        retset = retset / len(df.values)
        itemset = np.array(checkset)[retset > minSupport]
        supportset = retset[retset > minSupport]
        
        # 整理输出值
        for row, support in zip(itemset, supportset):
            strlst = [mapping[idx] if i > 0 else '' for idx, i in enumerate(row) if i == 1]
            data.append([support, ','.join(strlst)])
        
        # 两两合并，生成下一代迭代
        checkset.clear()
        tmp = set()
        for i in range(len(itemset)):
            for j in range(i + 1, len(itemset)):
                newitem = [1 if i >= 1 else 0 for i in itemset[i] + itemset[j]]
                key = "".join("{0}".format(n) for n in newitem)
                if not key in tmp:
                    checkset.append(newitem)
                    tmp.add(key)
        
    return pd.DataFrame(data, columns=["support", "items"]) 


def association_rules(df, minConf):
    supportset = {}
    for item in df.values:
        supportset[item[1]] = item[0]
    
    data = []    
    for row in df.values:
        items = row[1].split(',')
        if len(items) <= 1:
            continue
        
        # 排例组合，生成下一代迭代
        for i in range(2,len(items)):
            vv = list(combinations(items, i)) #排列组合
            for item in vv:
                items.append(",".join(item))
        
        for item in items:
            conf = row[0] / supportset[item]
            if conf > minConf:
                leitem = row[1].split(',')
                for s in item.split(','):
                    leitem.remove(s)
                sleitem = ",".join(leitem)
                left = conf / supportset[sleitem]
                data.append([item, sleitem, supportset[item], conf, left])
    
    return pd.DataFrame(data, columns=["antecedents", "consequents", "support", "confidence", "lift"]) 


if __name__ == '__main__':
    # 设置数据集
    dataset = [['牛奶', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
            ['莳萝', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
            ['牛奶', '苹果', '芸豆', '鸡蛋'],
            ['牛奶', '独角兽', '玉米', '芸豆', '酸奶'],
            ['玉米', '洋葱', '洋葱', '芸豆', '冰淇淋', '鸡蛋']]

    # 进行one-hot编码
    itemset = set()
    for row in dataset:
        for item in row:
            itemset.add(item);
            
    colnames = sorted(itemset)
    columns_mapping = {}
    for idx, col in enumerate(colnames):
        columns_mapping[col] = idx
#     print(columns_mapping)
    data = np.zeros((len(dataset), len(colnames)))
    for idx, row in enumerate(dataset):
        for item in row:
            data[idx, columns_mapping[item]] = 1
            
    df = pd.DataFrame(data, columns=colnames)    
#     print(df) 
     
    # 利用 Apriori 找出频繁项集
    freq = apriori(df, 0.5)
    print(freq)
    
    rule = association_rules(freq, 0.9)
    print(rule)

