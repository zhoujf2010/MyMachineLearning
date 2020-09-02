'''
Created on 2020年8月26日

@author: zjf

使用三方包直接使用aprior功能
'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

if __name__ == '__main__':
    #设置数据集
    dataset = [['牛奶','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'],
            ['莳萝','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'],
            ['牛奶','苹果','芸豆','鸡蛋'],
            ['牛奶','独角兽','玉米','芸豆','酸奶'],
            ['玉米','洋葱','洋葱','芸豆','冰淇淋','鸡蛋']]
            
    te = TransactionEncoder()
    #进行 one-hot 编码
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df)
    #利用 Apriori 找出频繁项集
    freq = apriori(df, min_support=0.6, use_colnames=True)
    print(freq)

    association_rule = association_rules(freq,metric='confidence',min_threshold=0.9)    # metric可以有很多的度量选项，返回的表列名都可以作为参数
#     association_rule.sort_values(by='leverage',ascending=False,inplace=True)    #关联规则可以按leverage排序
    print(association_rule.columns)
    print(association_rule[["antecedents","consequents","support","confidence","lift"]])
    
    
    