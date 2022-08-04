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
    '''들어온 단어를 spel_index number 로 변환하는 함수. 총 배열의 길이를 15로 맞추고 나머지 부분은 0으로 넣어준다. 

    Args:
        lst_word(str):  변환하고자 하는 str type의 단어
    
    Returns:
        index_word(list): spe_index number 로 변환된 list type의 단어
        None: 15글자를 초과하면 None
    '''
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
    '''단어에서 한 글자씩 뺀 것들을 반환하는 함수.

    Args:
        word(str): str type의 단어
    
    Returns:
        word_list(numpy.array): 한 글자씩 빠진 단어들의 numpy.array type의 배열
    '''
    word_list = []
    for i in range(len(word)):
        w = list(word)
        w.pop(i)
        w.append(" ")
        word_list.append(w)
    word_list = np.array(word_list)
    return word_list

def two_miss(word):
    '''단어에서 두 글자씩 뺀 것들을 반환하는 함수.

    Args:
        word(str): str type의 단어
    
    Returns:
        word_list(numpy.array): 두 글자씩 빠진 단어들의 numpy.array type의 배열
    '''
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
    '''단어에 랜덤한 위치에 랜덤한 한 글자 추가하는 노이즈 생성함수.
    
    Args:
        word_list(numpy.array): 노이즈를 추가할 numpy.array type의 단어 배열
    
    Returns:
        copied(numpy.array): 노이즈를 추가한 numpy.array type의 단어 배열
    '''
    copied = word_list.tolist()
    
    for i in range(0, len(copied)):
        x = randrange(0, len(copied[0]))
        y = randrange(0, 27)
        copied[i].insert(x, list(spel_index.keys())[y])
    copied = np.array(copied)
    return copied

def to_index(listed):
    '''list type의 단어 리스트를 spel_index number 로 변환해주는 함수

    Args:
        listed(list): list type의 단어 리스트

    Returns:
        indexed(numpy.array): spel_index number 로 변환된 단어 배열
    '''
    indexed = []
    for i in listed:
        indexed.append(indexing(i))
    indexed = np.array(indexed)
    return indexed
    
def series_to_list(text_word):
    """pandas.Series type의 단어 리스트를 list type으로 변경해주는 함수
    
    Args:
        text_word(pandas.Series): pandas.Series type의 단어 리스트
    
    Returns:
        listed(list): list type으로 변경된 단어 리스트
    """
    listed = []
    for i in text_word:
        listed.append(list(i))
    return listed

def load_others(path, word):
    '''others data를 불러오는 함수. trigger word로 설정한 단어는 삭제한다.

    Args:
        path(str): 불러올 데이터의 경로
        word(str): trigger word로 설정한 단어
    
    Returns:
        false_data(numpy.array): spel_index number로 변환된 단어 배열
    '''
    false = pd.read_csv(path,sep='\n', names=["words"])
    false.words = false.words.str.lower()

    if word in false.words.values:
        f_index = false.words[false.words == word].index[0]
        false_words = false.words.drop(f_index)
    else:
        false_words = false.words

    false_data = to_index(series_to_list(false_words))
    #false_label = np.zeros(shape=(false_data.shape[0],1),dtype=int)
    #false_data = np.hstack((false_data, false_label))
    return false_data

def change_spel(data, spel_from, spel_to):
    """기존 단어 리스트에서 spel(spel_index number 형태)을 변경해서 단어 리스트에 추가해주는 함수.
    변경시 중복되는 값이 있으면 제외시켜준다.

    Args:
        data(numpy.array): numpy.array type의 단어 리스트
        spel_from(str): from A to B 에서 A에 해당하는 spel
        spel_to(str): from A to B 에서 B에 해당하는 spel
    
    Returns:
        data(numpy.array): spel이 변경된 단어들을 추가받은 단어 리스트
    """
    changed = np.where(data == spel_index[spel_from], spel_index[spel_to], data)
    data = np.append(data, changed, axis=0)
    data = np.unique(data, axis=0)
    return data

def remove_overlap(true, false):
    """false 데이터와 true 데이터의 중복 값을 false 데이터에서 제거하는 함수.

    Args:
        true(numpy.array): augmentation 된 true 데이터
        false(numpy.array): false 데이터
    
    Returns:
        false(numpy.arry): true와의 중복값이 제거된 false 데이터
    """
    false = false.tolist()
    true = true.tolist()
    for i in true:
        if i in false:
            d_index = false.index(i)
            false.pop(d_index)
        
    false = np.array(false) 
    return false

def delete_blank(word):
    """zero padding으로 인한 오른쪽 공백을 없애주는 함수. index_spel 함수에서 사용된다.

    Args:
        word(list): str type을 요소로 가지는 list type의 단어 리스트
    
    Returns:
        w(list): str type을 요소로 가지는 list type의 단어 리스트
    """
    w = []
    for i in range(len(word)):
        w.append(word[i].rstrip())
    
    return w

def index_spel(data):
    """spel_index number로 변환되어있는 것을 문자로 변환해주는 함수.

    Args:
        data(numpy.array): spel_index number로 변환되어있는 단어 배열

    Returns:
        spel_data(list): str type을 요소로 가지는 list type의 단어 리스트 
    """
    spel_index_s = pd.Series(spel_index, dtype=np.int)
    list_data = spel_index_s.index
    list_name = spel_index_s.values
    num_to_spel = pd.Series(data = list_data, index = list_name)
    
    data_l = data.tolist()
    
    for i in data_l:
        for j in range(len(i)):
            i[j] = num_to_spel[i[j]]
    spel_data = []
    for k in data_l:
        spel_data.append(''.join(k))
    
    spel_data = delete_blank(spel_data)
    return spel_data

def create_dataset(other_path, c2v_model, word='friend'):
    """모델 학습을 위한 dataset을 생성해주는 함수. True data에는 trigger word로 설정된 단어를 augmentation 하고, 
    false data에는 true data와 중복된 값을 제거한다.
    
    Args:
        other_path(str): others 데이터를 가져오기 위한 경로
        c2v_model(model): load된 chars2vec 모델
        word(str): trigger word로 설정된 단어

    Returns:
        X_train(numpy.array): chars2vec으로 변환된 X_train 데이터
        y_train(numpy.array): X_train에 대한 label 값
    """
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
    
    others_x = load_others(other_path, word)
    others_x = remove_overlap(x, others_x)  
    
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
    """SVM model을 업데이트 하는 함수
    
    Args:
        trigger(str): trigger word로 설정된 단어
        c2v_model(model): load된 chars2vec 모델
        other_path(str): others 데이터를 가져오기 위한 경로
   
    Returns:
        svm_model(model): 새로운 trigger word로 업데이트 된 model
    """
    print("Updated Trigger: ", trigger)
    X_train, y_train = create_dataset(other_path, c2v_model, word=trigger)
    svm_model = SVC(kernel='rbf', C=8, gamma=0.1)
    svm_model.fit(X_train, y_train)
    return svm_model



