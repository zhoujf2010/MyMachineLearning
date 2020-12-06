'''
Created on 2020年11月19日

@author: zjf
'''
import numpy as np
from scipy.io import wavfile
import simpleaudio as sa

from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from keras.layers import GRU, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def insert_audio_clip(background, audio_clip, pos):
    """
    在pos处，将audio插入到background中
    """
    p = len(audio_clip)
    newdt = np.vstack((background[:pos] , audio_clip , background[pos + p:]))
    return newdt


def get_random_time_segment(cliplen, previous_segments):
    """
        取得一个随机位置,cliplen表示要插入的长度，防止到最尾头
    """
    while True:
        segment_start = np.random.randint(low=0, high=441000 - cliplen)  # 441000为10秒的背景音乐的长度，不能超过 
        segment_end = segment_start + cliplen - 1
        
        overlap = False
        for previous_start, previous_end in previous_segments:
            if segment_start <= previous_end and segment_end >= previous_start:
                overlap = True
        if not overlap:
            break
    previous_segments.append((segment_start, segment_end))
    return (segment_start, segment_end)


def calcSpectral(x, nfft, fs, stride):
    '''
    计算频谱信息
    '''
    windownum = (x.shape[-1] - nfft + stride) // stride
    numFreqs = nfft // 2 + 1
    
    # 滑动取出x的值
    strides = (x.strides[0], stride * x.strides[0])
    result = np.lib.stride_tricks.as_strided(x, shape=(nfft, windownum), strides=strides)

    # 滑动xxx
    tmp = np.ones(nfft, dtype=x.dtype)
    xt = np.hanning(nfft) * tmp
    strides = (xt.strides[0], 0)
    windowValsRep = np.lib.stride_tricks.as_strided(xt, shape=(nfft, windownum), strides=strides)  
    result = windowValsRep * result
    
    result = np.fft.fft(result, n=nfft, axis=0)[:numFreqs, :]
    result = np.conj(result) * result
    result = result.real
    slc = slice(1, -1, None)
    result[slc] *= 2
    result /= fs
    result /= (np.abs(xt) ** 2).sum()
    
    return result


def create_onetraining_example(backgrounds, activates, negatives, Ty):
    """
    创建一条训练数据
    """
    y = np.zeros((1, Ty))  # 定义输出y
 
    data = backgrounds[np.random.randint(0, len(backgrounds))]
    previous_segments = []
 
    # 随机挑选0~4个正样本，插入到背景
    random_indices = np.random.randint(len(activates), size=np.random.randint(1, 5))
    for index in random_indices:
        # 随机位置，并插入背景声音中
        insertpos = get_random_time_segment(len(activates[index]), previous_segments)
        data = insert_audio_clip(data, activates[index], insertpos[0])
        
        # 将插入点的尾部，记录y值为1
        segment_end_y = insertpos[1] * Ty // 441000
        for i in range(segment_end_y + 1, min(segment_end_y + 51, Ty)):
            y[0, i] = 1.0
 
    # 随机挑选0~2个负样本插入到背景中
    random_indices = np.random.randint(len(negatives), size=np.random.randint(0, 3))
    for index in random_indices:
        # 随机位置，并插入背景声音中
        insertpos = get_random_time_segment(len(negatives[index]), previous_segments)
        data = insert_audio_clip(data, negatives[index], insertpos[0])
 
    # 取得频域数据
    data = calcSpectral(data[:, 0], nfft, fs, stride)
 
    return data.T, y.T


def model(input_shape):
    """
    Function creating the model's graph in Keras.
 
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
 
    Returns:
    model -- Keras model instance
    """
 
    X_input = Input(shape=input_shape)
 
    ### START CODE HERE ###
 
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, 15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)
 
    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True, reset_after=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
 
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True, reset_after=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)
 
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)
 
    ### END CODE HERE ###
 
    model = Model(inputs=X_input, outputs=X)
 
    return model  


def detect_triggerword(filename, model):
    
    _, audio = wavfile.read(filename)
    audio_spec = calcSpectral(audio[:, 0], nfft, fs, stride)
     
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(np.flipud(10. * np.log10(audio_spec)))
    ax1.axis('auto')
    
    predictions = model.predict(np.array([audio_spec.T]))
 
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(predictions[0, :, 0])
#     ax2.ylabel('probability')
    
    plt.show()

# chime_file = "data/soundData/audio_examples/chime.wav"
# def chime_on_activate(filename, predictions, threshold):
#     audio_clip = AudioSegment.from_wav(filename)
#     chime = AudioSegment.from_wav(chime_file)
#     Ty = predictions.shape[1]
#     # Step 1: Initialize the number of consecutive output steps to 0
#     consecutive_timesteps = 0
#     # Step 2: Loop over the output steps in the y
#     for i in range(Ty):
#         # Step 3: Increment consecutive output steps
#         consecutive_timesteps += 1
#         # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
#         if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
#             # Step 5: Superpose audio and background using pydub
#             audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
#             # Step 6: Reset consecutive output steps to 0
#             consecutive_timesteps = 0
#  
#     audio_clip.export("chime_output.wav", format='wav')

if __name__ == '__main__':
    np.random.seed(5)

    rate, audio = wavfile.read("./data/soundData/activates/1.wav")
    print('rate=', rate, 'len=', np.shape(audio))
#     sa.play_buffer(audio, 2, 2, rate).wait_done()
    _, background = wavfile.read("./data/soundData/backgrounds/1.wav")
    
    newdt = insert_audio_clip(background, audio, 220000)
    print(np.shape(newdt))
#     sa.play_buffer(newdt, 2, 2, 44100).wait_done()
    # wavfile.write("test.wav", 44100, newdt)
    previous_segments = []
    p1 = get_random_time_segment(770, previous_segments)
    print(p1)
    p2 = get_random_time_segment(770, previous_segments)
    print(p2)

    nfft = 200  # 每个窗口段的大小
    stride = 80  # 窗口滑动步幅
    fs = 8000  # Sampling frequencies

    Tx = 5511  # (441000-200+80)/80 The number of time steps input to the model from the spectrogram
    n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375  # The number of time steps in the output of our model
    
    print(np.shape(background))
    background_spec = calcSpectral(background[:, 0], nfft, fs, stride)
    print(np.shape(background_spec))
    newdt_spec = calcSpectral(newdt[:, 0], nfft, fs, stride)
    audio_spec = calcSpectral(audio[:, 0], nfft, fs, stride)
     
#     ax1 = plt.subplot(2, 2, 1)
#     ax1.imshow(np.flipud(10. * np.log10(background_spec)))
#     ax1.axis('auto')
#     ax2 = plt.subplot(2, 2, 2)
#     ax2.imshow(np.flipud(10. * np.log10(audio_spec)))
#     ax2.axis('auto')
#     ax3 = plt.subplot(2, 1, 2)
#     ax3.imshow(np.flipud(10. * np.log10(newdt_spec)))
#     ax3.axis('auto')
     
#     plt.show()

    # 加载样本数据
    filepath = "./data/soundData"
    activates = [wavfile.read("%s/activates/%d.wav" % (filepath, i))[1] for i in range(0, 9)]
    backgrounds = [wavfile.read("%s/backgrounds/%d.wav" % (filepath, i))[1] for i in range(0, 1)]
    negatives = [wavfile.read("%s/negatives/%d.wav" % (filepath, i))[1] for i in range(0, 9)]
    
    # 创建一条数据
    x, y = create_onetraining_example(backgrounds, activates, negatives, Ty)
    
    print("shape(y)=", np.shape(x))
    print("shape(y)=", np.shape(y))
    
    # 创建多条训练数据
    X = []
    Y = []
    for _ in range(26):
        x, y = create_onetraining_example(backgrounds, activates, negatives, Ty)
        X.append(x)
        Y.append(y)
    
    X = np.array(X)
    Y = np.array(Y)
    print(np.shape(X))
    print(np.shape(Y))
     
    model = model(input_shape=(Tx, n_freq))
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary()
    
    model = load_model('./data/tr_model.h5')

#     model.fit(X, Y, batch_size=5, epochs=1)
#     model.save("./data/tr_model.h5");
    
    # 创建测试数据并验证
    X_dev = []
    Y_dev = []
    for _ in range(5):
        x, y = create_onetraining_example(backgrounds, activates, negatives, Ty)
        X_dev.append(x)
        Y_dev.append(y)
    
    X_dev = np.array(X_dev)
    Y_dev = np.array(Y_dev)
    loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)
    
    predictions = model.predict(np.array([X_dev[0]]))
    

    detect_triggerword("data/soundData/dev/1.wav", model)
#     chime_on_activate(filename, prediction, 0.5)
  