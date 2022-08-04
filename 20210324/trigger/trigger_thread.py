"""
@author: Mozilla
    Edited by Kojungbeom

reference:
    https://github.com/mozilla/DeepSpeech-examples/blob/r0.9/mic_vad_streaming/mic_vad_streaming.py
"""

import numpy as np
import pandas as pd
from random import *
from sklearn.svm import SVC
import chars2vec

spel_index = {' ':0, 'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,\
              'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}


def indexing(lst_word):
    if len(lst_word) < 16:
        index_word = np.zeros(shape=(15,),dtype=int)
        index_word = index_word.tolist()
        j = 0
        for i in lst_word:
            index_word[j] = spel_index[i]
            j += 1
        return index_word
    else:
        print("input under word length 15")
        return None

def one_miss(word):
    word_list = []
    for i in range(len(word)):
        w = list(word)
        w.pop(i)
        w.append(" ")
        word_list.append(w)
    word_list = np.array(word_list)
    return word_list

def two_miss(word):
    word_list = []
    r = int(len(word) * (len(word) - 1) / 2)    # nC2
    
    while len(word_list) < r:
        w = list(word)
        a = randrange(0, len(word))
        b = randrange(0, len(word) - 1)
        w.pop(a)
        w.pop(b)
        if w not in word_list:    
            word_list.append(w)
    word_list = np.array(word_list)
    #print(len(word_list))
    return word_list

def make_noise(word_list):
    copied = word_list.tolist()
    
    for i in range(0, len(copied)):
        x = randrange(0, len(copied[0]))
        y = randrange(0, 27)
        copied[i].insert(x, list(spel_index.keys())[y])
    copied = np.array(copied)
    return copied

def to_index(listed):
    indexed = []
    for i in listed:
        indexed.append(indexing(i))
    indexed = np.array(indexed)
    return indexed
    
def to_list(text_word):
    listed = []
    for i in text_word:
        listed.append(list(i))
    return listed

def load_others(path):
    false = pd.read_csv(path,sep='\n')
    false.word = false.word.str.lower()

    if "friend" in false.word.values:
        f_index = false.word[false.word == word].index[0]
        false_word = false.word.drop(f_index)
    else:
        false_word = false.word

    false_data = to_index(to_list(false_word))
    #false_label = np.zeros(shape=(false_data.shape[0],1),dtype=int)
    #false_data = np.hstack((false_data, false_label))
    return false_data

def change_spel(data, spel_from, spel_to):
    changed = np.where(data == spel_index[spel_from], spel_index[spel_to], data)
    data = np.append(data, changed, axis=0)
    data = np.unique(data, axis=0)
    return data

def remove_overlap(true, false):
    false = false.tolist()
    true = np.delete(true, -1, 1).tolist()
    for i in true:
        if i in false:
            d_index = false.index(i)
            false.pop(d_index)
        
    false = np.array(false) 
    return true, false

def delete_blank(word):  # input : list 형태
    w = []
    for i in range(len(word)):
        w.append(word[i].rstrip())
    
    return w

def index_spel(data):
    spel_index_s = pd.Series(spel_index, dtype=np.int)
    list_data = spel_index_s.index
    list_name = spel_index_s.values
    num_to_spel = pd.Series(data = list_data, index = list_name)
    
    data_l = data.tolist()
    
    for i in data_l:
        for j in range(len(i)):
            i[j] = num_to_spel[i[j]]
    spel_data = []
    for j in data_l:
        spel_data.append(''.join(j))
    
    spel_data = delete_blank(spel_data)
    return spel_data

def create_dataset(other_path, c2v_model, word='friend'):
    #print(word)
    word_list = np.array([list(word)])
    word_list = np.vstack((word_list, one_miss(word)))
    indexed_d = to_index(word_list)
    indexed_n = to_index(make_noise(word_list))

    if len(word) < 6:
        d_t_data  = indexed_d
    else:
        indexed_t = to_index(two_miss(word))
        d_t_data = np.vstack((indexed_d, indexed_t))

    #  모음 변환 a (1), e (5), i(9), o(15), u(21)
    change_a_e = change_spel(d_t_data, "a", "e")
    change_a_i = change_spel(d_t_data, "a", "i")
    change_a_o = change_spel(d_t_data, "a", "o")
    change_a_u = change_spel(d_t_data, "a", "u")
    change_a = np.vstack((change_a_e, change_a_i, change_a_o, change_a_u))

    change_e_a = change_spel(d_t_data, "e", "a")
    change_e_i = change_spel(d_t_data, "e", "i")
    change_e_o = change_spel(d_t_data, "e", "o")
    change_e_u = change_spel(d_t_data, "e", "u")
    change_e = np.vstack((change_e_a, change_e_i, change_e_o, change_e_u))
    
    change_i_a = change_spel(d_t_data, "i", "a")
    change_i_e = change_spel(d_t_data, "i", "e")
    change_i_o = change_spel(d_t_data, "i", "o")
    change_i_u = change_spel(d_t_data, "i", "u")
    change_i = np.vstack((change_i_a, change_i_e, change_i_o, change_i_u))

    change_o_a = change_spel(d_t_data, "o", "a")
    change_o_e = change_spel(d_t_data, "o", "e")
    change_o_i = change_spel(d_t_data, "o", "i")
    change_o_u = change_spel(d_t_data, "o", "u")
    change_o = np.vstack((change_o_a, change_o_e, change_o_i, change_o_u))

    change_u_a = change_spel(d_t_data, "u", "a")
    change_u_e = change_spel(d_t_data, "u", "e")
    change_u_i = change_spel(d_t_data, "u", "i")
    change_u_o = change_spel(d_t_data, "u", "o")
    change_u = np.vstack((change_u_a, change_u_e, change_u_i, change_u_o))
    
    d_t_data = np.vstack((change_a, change_e, change_i, change_o, change_u))
    d_t_data = np.unique(d_t_data, axis=0)
    
    x = np.append(d_t_data, indexed_n,axis =0)
    #y = np.ones(shape=(x.shape[0],1),dtype=int)
    #xy = np.hstack((x, y))
    
    others_x = load_others(other_path)
    
    
    w_t = index_spel(x)
    w_f = index_spel(others_x)

    #c2v_model = chars2vec.load_model('eng_50')

    true_word_embeddings = c2v_model.vectorize_words(w_t)
    false_word_embeddings = c2v_model.vectorize_words(w_f)

    true_label = np.ones(shape=(true_word_embeddings.shape[0],1),dtype=int)
    true_data = np.hstack((true_word_embeddings, true_label))

    false_label = np.zeros(shape=(false_word_embeddings.shape[0],1),dtype=int)
    false_data = np.hstack((false_word_embeddings, false_label))

    np.random.shuffle(true_data)
    np.random.shuffle(false_data)
    
    train_data = np.vstack((true_data, false_data))
       
    X_train = np.delete(train_data,-1,1)
    y_train = train_data[:,-1:].ravel()

    return X_train, y_train

def get_updated_model(trigger, c2v_model, other_path):
    print("Updated Trigger: ", trigger)
    X_train, y_train = create_dataset(other_path, c2v_model, word=trigger)
    svm_model = SVC(kernel='rbf', C=8, gamma=0.1)
    svm_model.fit(X_train, y_train)
    return svm_model



