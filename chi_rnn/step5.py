# -*- coding:utf-8 -*-
'''
Created on 2019年2月10日

@author: zjf
'''

'''
用LSTM即兴创作Jazz
#https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/tree/master/Sequence%20Models/week1/LSTM%20Network
'''


from music21 import stream, midi, tempo, note
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from chi_rnn import step5_preprocess as pps

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to sample an index from a probability array '''


def __sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


''' Helper function to generate a predicted value from a given matrix '''


def __predict(model, x, indices_val, diversity):
    preds = model.predict(x, verbose=0)[0]
    next_index = __sample(preds, diversity)
    next_val = indices_val[next_index]

    return next_val

''' Helper function which uses the given model to generate a grammar sequence 
    from a given corpus, indices_val (mapping), abstract_grammars (list), 
    and diversity floating point value. '''


def __generate_grammar(model, corpus, abstract_grammars, values, val_indices,
                       indices_val, max_len, max_tries, diversity):
    curr_grammar = ''
    # np.random.randint is exclusive to high
    start_index = np.random.randint(0, len(corpus) - max_len)
    sentence = corpus[start_index: start_index + max_len]  # seed
    running_length = 0.0
    while running_length <= 4.1:  # arbitrary, from avg in input file
        # transform sentence (previous sequence) to matrix
        x = np.zeros((1, max_len, len(values)))
        for t, val in enumerate(sentence):
            if (not val in val_indices): print(val)
            x[0, t, val_indices[val]] = 1.

        next_val = __predict(model, x, indices_val, diversity)

        # fix first note: must not have < > and not be a rest
        if (running_length < 0.00001):
            tries = 0
            while (next_val.split(',')[0] == 'R' or 
                len(next_val.split(',')) != 2):
                # give up after 1000 tries; random from input's first notes
                if tries >= max_tries:
                    print('Gave up on first note generation after', max_tries,
                        'tries')
                    # np.random is exclusive to high
                    rand = np.random.randint(0, len(abstract_grammars))
                    next_val = abstract_grammars[rand].split(' ')[0]
                else:
                    next_val = __predict(model, x, indices_val, diversity)

                tries += 1

        # shift sentence over with new value
        sentence = sentence[1:] 
        sentence.append(next_val)

        # except for first case, add a ' ' separator
        if (running_length > 0.00001): curr_grammar += ' '
        curr_grammar += next_val

        length = float(next_val.split(',')[1])
        running_length += length

    return curr_grammar


def data_processing(corpus, tones_indices, max_len, step=3):
#     number of different values or words in corpus
    N_values = len(set(corpus))
 
    # cut the corpus into semi-redundant sequences of max_len values
    sentences = []
    next_values = []
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print('nb sequences:', len(sentences))
 
    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, tones_indices[val]] = 1
        y[i, tones_indices[next_values[i]]] = 1
    return X, y


if __name__ == '__main__':
    # 加载midi数据
    chords, abstract_grammars = pps.get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = pps.get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    max_len = 20
    X, Y = data_processing(corpus, tones_indices, max_len, 3)   
    print('shape of X:', X.shape)
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', N_tones)
    print('Shape of Y:', Y.shape)

    max_tries = 1000
    diversity = 0.5
    N_epochs = 2
    
    # build model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, N_tones)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(N_tones))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()

    model.fit(X, Y, batch_size=128, epochs=N_epochs)
    
    # set up audio stream
    out_stream = stream.Stream()

    # 循环生成
    curr_offset = 0.0
    for loopIndex in range(1, len(chords)):
        # get chords from file
        curr_chords = stream.Voice()
        for j in chords[loopIndex]:
            curr_chords.insert((j.offset % 4), j)

        # generate grammar
        curr_grammar = __generate_grammar(model=model, corpus=corpus,
                                          abstract_grammars=abstract_grammars,
                                          values=tones, val_indices=tones_indices,
                                          indices_val=indices_tones,
                                          max_len=max_len, max_tries=max_tries,
                                          diversity=diversity)

        curr_grammar = curr_grammar.replace(' A', ' C').replace(' X', ' C')
        # Pruning #1: smoothing measure
        curr_grammar = pps.prune_grammar(curr_grammar)
        # Get notes from grammar and chords
        curr_notes = pps.unparse_grammar(curr_grammar, curr_chords)
        # Pruning #2: removing repeated and too close together notes
        curr_notes = pps.prune_notes(curr_notes)
        # quality assurance: clean up notes
        curr_notes = pps.clean_up_notes(curr_notes)
        # print # of notes in curr_notes
        print('After pruning: %s notes' % (len([i for i in curr_notes
            if isinstance(i, note.Note)])))

        # insert into the output stream
        for m in curr_notes:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # musical settings
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    #播放生成的音乐
    midi.realtime.StreamPlayer(out_stream).play()

    #保存文件
#     mf = midi.translate.streamToMidiFile(out_stream)
#     mf.open("out.mid", 'wb')
#     mf.write()
#     mf.close()