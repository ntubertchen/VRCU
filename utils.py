import json
import numpy as np
import h5py
import re
import sys
import itertools
import pandas as pd
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"(\-)+>|(\=)+>", "=> ", string) # '-->', '=>', '==>' are all maped to '=>'
    string = re.sub(r"[^A-Za-z0-9()\+\-\'/]", " ", string)  # Note: Chinese characters are not included
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\(", "   ", string)
    string = re.sub(r"\)", "   ", string)
    string = re.sub(r"\+", " + ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"/", " / ", string)
    string = re.sub(r"n't", " not ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def one_hot_object(target_object_id,object_list):
    no_object = True
    obj_np = np.zeros(94)
    for obj in object_list:
        if obj['id'] == target_object_id:
            obj['category_id'] -= 1
            no_object = False
            assert obj['category_id'] < 90
            assert obj['category_id'] >= 0
            obj_np[obj['category_id']] = 1
            obj_np[80] = obj['bbox'][0]#x
            obj_np[81] = obj['bbox'][1]#y
            obj_np[82] = obj['bbox'][2]#w
            obj_np[83] = obj['bbox'][3]#h
    assert no_object == False
    return obj_np

def load_data(path):
    # utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl')
    #'/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl'
    vocabulary, vocabulary_inv = build_vocab()
    file_list = open(path,'r')
    image_to_float = []
    question = []
    answer = []
    target_object_list = []
    arxiv = {}
    count = 0
    feature_map_index_to_question = []
    mapping_index = 0
    for line in file_list:
        data_info = json.loads(line)
        image_name = data_info['image']['file_name']
        qa_list = data_info['qas'] #dict answer question
        target_object = data_info['object_id']
        object_list = data_info['objects']
        target_object_np = one_hot_object(target_object,object_list)
        for qa_pair in qa_list:
            mapping_index += 1
            image_to_float.append(count)
            question.append(clean_str(qa_pair['question']).split(' '))
            answer.append(qa_pair['answer'])
            target_object_list.append(target_object_np)
        count += 1
        feature_map_index_to_question.append(mapping_index)
        arxiv[image_name] = {'qa_list':qa_list,'target_object':target_object,'object_list':object_list}
    answer = transform_ans_to_onehot(answer)
    print ('answer transform done')
    pretrained_embedding = load_word_embedding()

    question = trainsform_word_to_index(question,pretrained_embedding,vocabulary)
    embedding_weights = word2vec_to_index2vec(pretrained_embedding, vocabulary_inv)

    return feature_map_index_to_question, image_to_float, np.array(question), np.array(answer), np.array(target_object_list), arxiv, embedding_weights

def trainsform_word_to_index(question,pretrained_embedding,vocabulary):
    question_index = []
    for q in question:
        q_index = []
        for i in range(min(len(q),20)):
            q_index.append(vocabulary[q[i]])
        if len(q_index) < 20:#max length
            q_index += [vocabulary["<PAD/>"]] * (20 - len(q_index))    
        question_index.append(q_index)
    return question_index

def transform_ans_to_onehot(answer):
    ans = []
    for a in answer:
        temp = [0] * 2
        if a == 'Yes':
            temp[0] = 1
        elif a == 'No':
            temp[1] = 1
        elif a == 'N/A':
            temp[0] = 1
        else:
            print ('weird answer',a)
        ans.append(temp)
    return ans

def word2vec_to_index2vec(pretrained_embedding,vocabulary_inv):
    OOVf = open('oov.txt','w')
    embedding_weights = {}
    for id, word in vocabulary_inv.items():
        if word in pretrained_embedding.vocab:
            embedding_weights[id] = pretrained_embedding.word_vec(word)
        else:
            OOVf.write(word+'\n')
            embedding_weights[id] = np.random.uniform(-0.25, 0.25, 300)

    OOVf.close()
    embedding_weights_list = []
    for i in range(len(embedding_weights.keys())):
        embedding_weights_list.append(embedding_weights[i])
    return np.array(embedding_weights_list)

def load_word_embedding():
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    # glove_input_file = "glove.txt"# glove path
    word2vec_output_file = '/tmp2/glove/pretrained_6_embedding.txt' # word2vec path
    # glove2word2vec(glove_input_file, word2vec_output_file)
    pretrained_embedding = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    return pretrained_embedding

def build_vocab():
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    file_list = open('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl','r')
    question = []
    for line in file_list:
        data_info = json.loads(line)
        qa_list = data_info['qas'] #dict answer question
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
    file_list.close()
    file_list = open('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl','r')
    for line in file_list:
        data_info = json.loads(line)
        qa_list = data_info['qas'] #dict answer question
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
    file_list.close()
    # Build vocabulary
    temp = question + [['<PAD/>']]
    word_counts = Counter(itertools.chain(*temp))
    # Mapping from index to word
    indexing = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(indexing)}
    vocabulary_inv = {i:x for i, x in enumerate(indexing)}
    return [vocabulary, vocabulary_inv]

def load_feature_map(path):
    #'/tmp2/train_nas_h5/train.hdf5'
    output = open(path,'r')
    feature_map = output['train'][:]
    return feature_map
