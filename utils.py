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

def one_hot_object(target_object_id,object_list,image):
    no_object = True
    obj_np = np.zeros(98)
    i_w = image['width']
    i_h = image['height']
    for obj in object_list:
        if obj['id'] == target_object_id:
            obj['category_id'] -= 1
            no_object = False
            assert obj['category_id'] < 90
            assert obj['category_id'] >= 0
            obj_np[obj['category_id']] = 1
            obj_np[90] = (obj['bbox'][0]-(float(obj['bbox'][2])/2))/float(i_w)#x min
            obj_np[91] = (obj['bbox'][1]-(float(obj['bbox'][3])/2))/float(i_h)#y min
            obj_np[92] = (obj['bbox'][0]+(float(obj['bbox'][2])/2))/float(i_w)#x max
            obj_np[93] = (obj['bbox'][1]+(float(obj['bbox'][3])/2))/float(i_h)#y max
            obj_np[94] = obj['bbox'][0]/float(i_w) #x center
            obj_np[95] = obj['bbox'][1]/float(i_h) #y center
            obj_np[96] = obj['bbox'][2]/float(i_w)#
            obj_np[97] = obj['bbox'][3]/float(i_h)#h
    assert no_object == False
    return obj_np

def ground_truth_feature(target_object_id,object_list,image):
    no_object = True
    gt = np.zeros((20,98))
    target_object_np = np.zeros(98)
    i_w = image['width']
    i_h = image['height']
    count = 0
    for obj in object_list:
        obj['category_id'] -= 1
        no_object = False
        assert obj['category_id'] < 90
        assert obj['category_id'] >= 0
        gt[count][obj['category_id']] = 1
        gt[count][90] = (obj['bbox'][0]-(float(obj['bbox'][2])/2))/float(i_w)#x min
        gt[count][91] = (obj['bbox'][1]-(float(obj['bbox'][3])/2))/float(i_h)#y min
        gt[count][92] = (obj['bbox'][0]+(float(obj['bbox'][2])/2))/float(i_w)#x max
        gt[count][93] = (obj['bbox'][1]+(float(obj['bbox'][3])/2))/float(i_h)#y max
        gt[count][94] = obj['bbox'][0]/float(i_w) #x center
        gt[count][95] = obj['bbox'][1]/float(i_h) #y center
        gt[count][96] = obj['bbox'][2]/float(i_w)#
        gt[count][97] = obj['bbox'][3]/float(i_h)#h     
        if obj['id'] == target_object_id:
            target_object_np[obj['category_id']] = 1
            target_object_np[90] = (obj['bbox'][0]-(float(obj['bbox'][2])/2))/float(i_w)#x min
            target_object_np[91] = (obj['bbox'][1]-(float(obj['bbox'][3])/2))/float(i_h)#y min
            target_object_np[92] = (obj['bbox'][0]+(float(obj['bbox'][2])/2))/float(i_w)#x max
            target_object_np[93] = (obj['bbox'][1]+(float(obj['bbox'][3])/2))/float(i_h)#y max
            target_object_np[94] = obj['bbox'][0]/float(i_w) #x center
            target_object_np[95] = obj['bbox'][1]/float(i_h) #y center
            target_object_np[96] = obj['bbox'][2]/float(i_w)#
            target_object_np[97] = obj['bbox'][3]/float(i_h)#h
        count += 1
    assert no_object == False
    return target_object_np, gt, count

def experiment_load_data(file_name,jsonfile,args):
    vocabulary, vocabulary_inv = build_vocab()
    f = open(jsonfile,'r')
    for l in f:
        image_to_index = json.loads(l)
    print ('image len',len(image_to_index))
    f.close()
    file_list = open(file_name,'r')
    question = []
    answer = []
    target_object_list = []
    arxiv = {}
    feature_map_index_list = []
    count = 0
    obj_in_one_image_list = []
    ground_truth_feature_list = []
    for line in file_list:
        count += 1
        data_info = json.loads(line)
        # if 'train' in file_name and args.use_simple:
        #     if data_info['status'] != 'success':
        #         continue
        image_name = data_info['image']['file_name']
        if 'train' in image_name:
            feature_map_index = image_to_index['/tmp2/train2014/'+image_name]
        elif 'val' in image_name:
            feature_map_index = image_to_index['/tmp2/val2014/'+image_name]
        else:
            print ('image_name error',image_name)
        qa_list = data_info['qas'] #dict answer question
        target_object = data_info['object_id']
        object_list = data_info['objects']
        target_object_np, gt_feature,obj_in_one_image = ground_truth_feature(target_object,object_list,data_info['image']) #target gt count
        obj_in_one_image_list.append(obj_in_one_image)
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
            answer.append(qa_pair['answer'])
            target_object_list.append(target_object_np)
            ground_truth_feature_list.append(gt_feature)
        arxiv[image_name] = {'qa_list':qa_list,'target_object':target_object,'object_list':object_list}
    answer = transform_ans_to_onehot(answer)
    print ('answer transform done, question set:',count)
    pretrained_embedding = load_word_embedding()

    question = trainsform_word_to_index(question,pretrained_embedding,vocabulary)
    embedding_weights = word2vec_to_index2vec(pretrained_embedding, vocabulary_inv)

    return np.array(ground_truth_feature_list), np.array(question), np.array(answer), np.array(target_object_list), arxiv, embedding_weights, obj_in_one_image_list

def print_wa(file_name,wa):
    file_list = open(file_name,'r')
    question = []
    answer = []
    object_list = []
    target_id_list = []
    for line in file_list:
        data_info = json.loads(line)
        qa_list = data_info['qas'] #dict answer question
        target_object = data_info['object_id']
        object_list = data_info['objects']
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
            answer.append(qa_pair['answer'])
            object_list.append(object_list)
            target_id_list.append(target_object)
    for idx in wa:
        print (question[idx],answer[idx],target_id_list[idx],object_list[idx])

def load_data(file_name,jsonfile):
    # utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl')
    #'/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl'
    vocabulary, vocabulary_inv = build_vocab()
    f = open(jsonfile,'r')
    for l in f:
        image_to_index = json.loads(l)
    print ('image len',len(image_to_index))
    f.close()
    file_list = open(file_name,'r')
    question = []
    answer = []
    target_object_list = []
    arxiv = {}
    feature_map_index_list = []
    count = 0
    for line in file_list:
        count += 1
        data_info = json.loads(line)
        image_name = data_info['image']['file_name']
        if 'train' in image_name:
            feature_map_index = image_to_index['/tmp2/train2014/'+image_name]
        elif 'val' in image_name:
            feature_map_index = image_to_index['/tmp2/val2014/'+image_name]
        else:
            print ('image_name error',image_name)
        qa_list = data_info['qas'] #dict answer question
        target_object = data_info['object_id']
        object_list = data_info['objects']
        target_object_np = one_hot_object(target_object,object_list,data_info['image'])
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
            answer.append(qa_pair['answer'])
            target_object_list.append(target_object_np)
            feature_map_index_list.append(feature_map_index)
        arxiv[image_name] = {'qa_list':qa_list,'target_object':target_object,'object_list':object_list}
    answer = transform_ans_to_onehot(answer)
    print ('answer transform done, question set:',count)
    pretrained_embedding = load_word_embedding()

    question = trainsform_word_to_index(question,pretrained_embedding,vocabulary)
    embedding_weights = word2vec_to_index2vec(pretrained_embedding, vocabulary_inv)

    return feature_map_index_list, np.array(question), np.array(answer), np.array(target_object_list), arxiv, embedding_weights

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
        temp = [0] * 3
        if a == 'Yes':
            temp[0] = 1
        elif a == 'No':
            temp[1] = 1
        elif a == 'N/A':
            temp[2] = 1
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
    file_list = open('/tmp2/val_nas_h5/guesswhat.valid.new.jsonl','r')
    question = []
    for line in file_list:
        data_info = json.loads(line)
        qa_list = data_info['qas'] #dict answer question
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
    file_list.close()
    file_list = open('/tmp2/test_nas_h5/guesswhat.test.jsonl','r')
    for line in file_list:
        data_info = json.loads(line)
        qa_list = data_info['qas'] #dict answer question
        for qa_pair in qa_list:
            question.append(clean_str(qa_pair['question']).split(' '))
    file_list.close()
    file_list = open('/tmp2/train_nas_h5/guesswhat.train.new.jsonl','r')
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
