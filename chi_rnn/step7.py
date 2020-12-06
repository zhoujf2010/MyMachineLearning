'''
Created on 2020年11月19日

@author: zjf

机器翻译示例（人类时间表达-》机器时间）
'''
import random

from babel.dates import format_date
from faker import Faker
from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from tqdm import tqdm

import keras.backend as K
import numpy as np


# from nmt_utils import *
def load_dataset(m):
    """
       生成训练数据[[人类时间，计算机时间]]
    """

    FORMATS = ['short', 'medium', 'long', 'full''d MMM YYY', 'd MMMM YYY', 'dd MMM YYY', 'd MM YY',
               'd MMMM YYY', 'MMMM d YYY', 'MMMM d, YYY', 'dd.MM.YY']
    
    fake = Faker()
    Faker.seed(1)
    random.seed(1)

    dataset = []
    h_vocab = set()  # 记录出现的字符
    m_vocab = set()  # 记录出现的字符
    for _ in tqdm(range(m)):
        dt = fake.date_object()  # 随机生成一个日期数据
        h = format_date(dt, format=random.choice(FORMATS), locale='en_US')  # locale=random.choice(LOCALES))
        h = h.lower()
        m = dt.isoformat()
        dataset.append((h, m))
        
        h_vocab.update(tuple(h))
        m_vocab.update(tuple(m))
    return dataset, h_vocab, m_vocab


def string_to_int(string, length, vocab):
    string = string.lower()
    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    return rep


def preprocess_data(dataset, h_vocab, m_vocab, Tx, Ty):
    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, h_vocab) for i in X])
    Y = [string_to_int(t, Ty, m_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(h_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(m_vocab)), Y)))

    return Xoh, Yoh


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
 
 
def model(Tx, Ty, n_a, n_s, h_vocab_size, m_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"
 
    Returns:
    model -- Keras model instance
    """
 
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, h_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
 
    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(softmax, name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=1)
    
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(m_vocab_size, activation=softmax)
 
    # Initialize empty list of outputs
    outputs = []
 
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
 
    # Step 2: Iterate for Ty steps
    for _ in range(Ty):
 
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        concat = concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas, a])
 
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
 
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
 
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
 
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model
 

if __name__ == '__main__':
    m = 100
    # 生成数据
    dataset, h_vocab, m_vocab = load_dataset(m)
    # 提取字典
    h_vocab = dict(zip(sorted(h_vocab) + ['<unk>', '<pad>'], list(range(len(h_vocab) + 2))))
    m_vocab = dict(zip(sorted(m_vocab), list(range(len(m_vocab)))))
    inv_m_vocab = {v:k for k, v in m_vocab.items()}
    print('february 22, 1979', '---->', string_to_int('february 22, 1979', 30, h_vocab))
    print('1979-02-22', '---->', string_to_int('1979-02-22', 10, m_vocab))
    
    Tx = 30  # 输入最长30字符
    Ty = 10  # 输出最长10字符
    Xoh, Yoh = preprocess_data(dataset, h_vocab, m_vocab, Tx, Ty)
    
    print(dataset[:10])
    print("Source date:", dataset[1][0])
    print("Target date:", dataset[1][1])
    print("Source after preprocessing (one-hot):", Xoh[1])
    print("Target after preprocessing (one-hot):", Yoh[1])
    
    #定义模型
    n_a = 32
    n_s = 64
    model = model(Tx, Ty, n_a, n_s, len(h_vocab), len(m_vocab))
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    #模型训练 
    outputs = list(Yoh.swapaxes(0, 1))
    model.fit([Xoh, np.zeros((m, n_s)), np.zeros((m, n_s))], outputs, epochs=2, batch_size=1000)
     
    #模型验证
    EXAMPLES = ['9 may 1998', '3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
     
        source = string_to_int(example, Tx, h_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(h_vocab)), source))).swapaxes(0, 1).T
    
        prediction = model.predict([source.reshape([1, 30, 37]), np.zeros((1, n_s)), np.zeros((1, n_s))])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_m_vocab[int(i)] for i in prediction]
     
        print("source:", example)
        print("output:", ''.join(output))
         
