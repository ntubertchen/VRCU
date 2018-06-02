import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as np
import h5py
import argparse
import sys

from model import Model 
import utils

def validation(model):
    valid_feature_to_question_index, valid_q, valid_a, valid_target, valid_arxiv, word_embedding = utils.load_data('/tmp2/val_nas_h5/guesswhat.valid.new.jsonl','/tmp2/val_nas_h5/image_to_idx.json')
    valid_feature = h5py.File('/tmp2/val_nas_h5/val_all.hdf5','r')
    valid_feature = valid_feature['all'][:]
    predict = []
    batch_size = 256
    for i in range(int(len(valid_q)/batch_size)):
        question = np.array(valid_q[i*batch_size:(i+1)*batch_size])
        target = np.array(valid_target[i*batch_size:(i+1)*batch_size])
        temp = []
        for j in range(i*batch_size,(i+1)*batch_size):
            temp.append(valid_feature_to_question_index[j])
        feature = Variable(torch.from_numpy(valid_feature[temp]).float()).cuda()
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,Variable(torch.from_numpy(target).float().cuda()))
        output = output.data.cpu().numpy()
        # if len(predict) == 0:
        #     predict = [output]
        # else:
            # predict = np.concatenate((np.array(predict),output),axis=0)
        predict.append(output)
    predict = np.array(predict).reshape(-1,3)
    if len(valid_q) % batch_size != 0:
        question = np.array(valid_q[int(len(valid_q)/batch_size)*batch_size:])
        target = np.array(valid_target[int(len(valid_q)/batch_size)*batch_size:])
        temp = []
        for j in range(int(len(valid_q)/batch_size)*batch_size,len(valid_q)):
            temp.append(valid_feature_to_question_index[j])
        feature = Variable(torch.from_numpy(valid_feature[temp]).float()).cuda()
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,Variable(torch.from_numpy(target).float()).cuda())
        output = output.data.cpu().numpy()
        predict = np.concatenate((predict,output),0)

    predict = np.argmax(predict,-1)
    answer = np.argmax(valid_a,-1)
    correct = 0
    for i in range(len(predict)):
        if predict[i] == answer[i]:
            correct += 1
    print ('validation acc',float(correct)/float(len(predict)))
    return float(correct)/float(len(predict))

def testing(model):
    test_feature_to_question_index, test_q, test_a, test_target, test_arxiv, word_embedding = utils.load_data('/tmp2/test_nas_h5/guesswhat.test.jsonl','/tmp2/test_nas_h5/image_to_idx.json')
    test_feature = h5py.File('/tmp2/test_nas_h5/test_all.hdf5','r')
    test_feature = test_feature['all'][:]
    predict = []
    batch_size = 256
    for i in range(int(len(test_q)/batch_size)):
        question = np.array(test_q[i*batch_size:(i+1)*batch_size])
        target = np.array(test_target[i*batch_size:(i+1)*batch_size])
        temp = []
        for j in range(i*batch_size,(i+1)*batch_size):
            temp.append(test_feature_to_question_index[j])
        feature = Variable(torch.from_numpy(test_feature[temp]).float()).cuda()
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,Variable(torch.from_numpy(target).float().cuda()))
        output = output.data.cpu().numpy()
        # if len(predict) == 0:
        #     predict = [output]
        # else:
            # predict = np.concatenate((np.array(predict),output),axis=0)
        predict.append(output)
    predict = np.array(predict).reshape(-1,3)
    if len(test_q) % batch_size != 0:
        question = np.array(test_q[int(len(test_q)/batch_size)*batch_size:])
        target = np.array(test_target[int(len(test_q)/batch_size)*batch_size:])
        temp = []
        for j in range(int(len(test_q)/batch_size)*batch_size,len(test_q)):
            temp.append(test_feature_to_question_index[j])
        feature = Variable(torch.from_numpy(test_feature[temp]).float()).cuda()
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,Variable(torch.from_numpy(target).float()).cuda())
        output = output.data.cpu().numpy()
        predict = np.concatenate((predict,output),0)

    predict = np.argmax(predict,-1)
    answer = np.argmax(test_a,-1)
    correct = 0
    for i in range(len(predict)):
        if predict[i] == answer[i]:
            correct += 1
    print ('testing acc',float(correct)/float(len(predict)))
    return float(correct)/float(len(predict))

def load_train_data():
    train_feature = h5py.File('/tmp2/train_nas_h5/train_all.hdf5','r')
    return train_feature['all'][:]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):

    torch.manual_seed(1000)
    train_question_to_feature_index, train_q, train_a, train_target, train_arxiv, word_embedding = utils.load_data('/tmp2/train_nas_h5/guesswhat.train.new.jsonl','/tmp2/train_nas_h5/image_to_idx.json')

    train_feature = load_train_data()
    

    model = Model(vocab_size=len(word_embedding),
        emb_dim=300,
        feature_dim=4032,
        hidden_dim=1000,
        out_dim=3,
        pretrained_embedding=word_embedding
        ).cuda()
    if args.use_pretrain:
         model.load_state_dict(torch.load('./test_lstm_model'))
    print ('model size',count_parameters(model))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_acc = 0
    BATCH_SIZE = 64
    for epoch in range(args.epochs):
        model.train()
        loss_record = []
        print ('feature_map length:',len(train_feature))
        print ('question length:',len(train_q))
        r = torch.from_numpy(np.array([j for j in range(len(train_q))]))
        torch_dataset = Data.TensorDataset(data_tensor=r,target_tensor=r)
        loader = Data.DataLoader(dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
            )
        for step, (x_index,_) in enumerate(loader):
            x_index = x_index.numpy()
            q = Variable(torch.from_numpy(train_q[x_index])).cuda()
            a = np.argmax(train_a[x_index],axis=-1)
            a = Variable(torch.from_numpy(a)).cuda()
            target = Variable(torch.from_numpy(train_target[x_index]).float()).cuda()
            temp = []
            for idx in x_index:
                temp.append(train_question_to_feature_index[idx])
            feature = Variable(torch.from_numpy(train_feature[temp]).float()).cuda()

            output = model(q,feature,target)
            loss = loss_function(output, a)
            loss_record.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 and step > 0:
                print ('epoch:',epoch,step,sum(loss_record)/len(loss_record))
                loss_record = []
        sys.stdout.flush()
        if True:
            model.eval()
            t_acc = testing(model)
            if t_acc > val_acc:
                val_acc = t_acc
                torch.save(model.state_dict(), './test_lstm_model')
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', metavar='', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--epochs', metavar='', type=int, default=30, help='number of epochs.')
    parser.add_argument('--use_pretrain', action='store_true')
    args, unparsed = parser.parse_known_args()
    _ = train(args)
