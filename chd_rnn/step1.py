# -*- coding:utf-8 -*-
'''
Created on 2018年8月1日

@author: zjf
'''
'''
用TF定义RNN，实现恐龙起名
'''

import os
import time

import numpy as np
import  tensorflow as tf


def loadData(filename):
    # 预处理文本数据
    text = open(filename).read().lower()
    print("文档总长度", len(text))
    vocab = sorted(set(text))
    print("包含词数：", len(vocab), "list:", vocab)
    char_to_ix = {ch:i for i, ch in enumerate(vocab)}
    ix_to_char = np.array(vocab)
    print('char_to_ix:', char_to_ix)
    print('ix_to_char:', ix_to_char)
    
    # 加载数据
    with open(filename) as f:
        examples = f.readlines()
        examples = [[char_to_ix[x] for x in row.lower().strip()] for row in examples]
        
    # 计算出最长可能的单词数
    maxlen = max([len(x) for x in examples]) + 1
    print('maxlen:', maxlen)
    # 按最大长度单词长度，不足的补充0(\n)
    rows = [np.pad(row, (0, maxlen), 'constant', constant_values=0)[:maxlen] for row in examples]
    print("row0:", rows[0])
    print("row1:", rows[1])
    return rows, char_to_ix, ix_to_char, vocab, maxlen


if __name__ == '__main__':
    np.set_printoptions(linewidth=500)
    tf.enable_eager_execution()
    
    rows, char_to_ix, ix_to_char, vocab, maxlen = loadData("dinos.txt")
    
    # 装入tensor中
    char_dataset = tf.data.Dataset.from_tensor_slices(rows)
    # 错位，生成X与Y
    dataset = char_dataset.map(lambda row:(row[:-1], row[1:]))
    print(type(dataset))
    for segX, segY in dataset.take(2):
        print("X:", repr(''.join(ix_to_char[segX.numpy()])))
        print("Y:", repr(''.join(ix_to_char[segY.numpy()])))
    
    # 数据打包成批
    Batch_size = 64
    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size).batch(Batch_size, drop_remainder=True)
    
    vocab_size = len(vocab)  # Length of the vocabulary in chars
    embedding_dim = 50  # The embedding dimension 
    rnn_units = 50  # Number of RNN units
      
    def createmodel(batchsize):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batchsize, None]),
            tf.keras.layers.GRU(rnn_units, return_sequences=True, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform", stateful=True),
            tf.keras.layers.Dense(vocab_size)
            ])
        return model
    
    # 定义训练模型
    model = createmodel(Batch_size)
    model.build(tf.TensorShape([Batch_size, maxlen]))
    model.summary()  # 输出描述信息
      
    optimizer = tf.train.AdamOptimizer()
  
    def lossfunction(rel, preds):
        return tf.losses.sparse_softmax_cross_entropy(rel, preds)
     
    EPOCHS = 5
    for epoch in range(EPOCHS):
        start = time.time()
        hidden = model.reset_states()
        for(batch, (inp, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(inp)
                loss = lossfunction(target, predictions)
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))
                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))
  
        print ('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
     
    # 保存模型
    checkpoint_dir = './training_checkpoints'
    model.save_weights(os.path.join(checkpoint_dir, "ckpt"))
 
    # #预测新词
    model = createmodel(1)  # 定义输入1维
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    # 定义启动词
    start_string = 'a'
    input_eval = [char_to_ix[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    temperature = 1.0
    text_generated = []
    # 递推式预测后面的词，总长不超过50
    for i in range(50):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        if predicted_id == 0:
            break  # 碰到\n 表示结束
        text_generated.append(ix_to_char[predicted_id])
    print (start_string + ''.join(text_generated))
