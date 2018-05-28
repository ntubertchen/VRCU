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

def valid(args):
    torch.manual_seed(1000)

    #train_featuremapping, train_image_idx, train_q, train_a, train_target, train_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl')
    _, valid_featuremapping, valid_image_name, valid_q, valid_a, valid_target, valid_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl')
    valid_feature = h5py.File('/tmp2/val_nas_h5/valid.hdf5')
    #train 113221

    model = Model(vocab_size=len(word_embedding),emb_dim=300,feature_dim=4032,hidden_dim=500,out_dim=2,pretrained_embedding=word_embedding,).cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    BATCH_SIZE = 32
    print ('valid len',len(valid_q))
    r = torch.from_numpy(np.array([i for i in range(len(valid_q))]))
    torch_dataset = Data.TensorDataset(data_tensor=r,target_tensor=r)
    loader = Data.DataLoader(dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
        )
    feature_map = valid_feature['train'][:]
    for epoch in range(args.epochs):
        loss_record = []
        for step, (x_index,_) in enumerate(loader):
            x_index = x_index.numpy()
            q = Variable(torch.from_numpy(valid_q[x_index])).cuda()
            a = np.argmax(valid_a[x_index],axis=-1)
            a = Variable(torch.from_numpy(a)).cuda()
            target = Variable(torch.from_numpy(valid_target[x_index]).float()).cuda()
            temp = []
            for idx in x_index:
                temp.append(valid_featuremapping[valid_image_name[idx]])
            feature = Variable(torch.from_numpy(feature_map[temp]).float()).cuda()

            output = model(q,feature,target)
            loss = loss_function(output, a)
            # print (loss)
            loss_record.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0 and step > 0:
                print (step,sum(loss_record)/len(loss_record))
                loss_record = []
        
        print ('finish one epoch:',epoch)
        if epoch % 3 == 0 and epoch >0:
            test(model,valid_q,feature_map,valid_target,valid_a,valid_featuremapping,valid_image_name)

def validation(model):
    _, valid_image_idx, valid_q, valid_a, valid_target, valid_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl')
    valid_feature = h5py.File('/tmp2/val_nas_h5/valid.hdf5')
    valid_feature = valid_feature['train'][:]
    predict = []
    batch_size = 32
    for i in range(int(len(valid_q)/batch_size)):
        question = np.array(valid_q[i*batch_size:(i+1)*batch_size])
        target = np.array(valid_target[i*batch_size:(i+1)*batch_size])
        temp = []
        for j in range(i*batch_size,(i+1)*batch_size):
            temp.append(valid_image_idx[j])
        feature = Variable(torch.from_numpy(valid_feature[temp]).float()).cuda()
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,Variable(torch.from_numpy(target).float().cuda()))
        output = output.data.cpu().numpy()
        if len(predict) == 0:
            predict = output
        else:
            predict = np.concatenate((np.array(predict),output),axis=0)
    if len(valid_q) % batch_size != 0:
        question = np.array(valid_q[int(len(valid_q)/batch_size)*batch_size:])
        target = np.array(valid_target[int(len(valid_q)/batch_size)*batch_size:])
        temp = []
        for j in range(int(len(valid_q)/batch_size)*batch_size,len(valid_q)):
            temp.append(valid_image_idx[j])
        feature = Variable(torch.from_numpy(valid_feature[temp]).float()).cuda()
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,
        Variable(torch.from_numpy(target).float()).cuda())
        output = output.data.cpu().numpy()
        predict = np.concatenate((predict,output),0)

    predict = np.argmax(predict,-1)
    answer = np.argmax(valid_a,-1)
    correct = 0
    for i in range(len(predict)):
        if predict[i] == answer[i]:
            correct += 1
    print ('testing acc',float(correct)/float(len(predict)))

def test(model,q,f,t,a,valid_featuremapping,valid_image_name):
    batch_size = 5
    predict = []
    for i in range(int(len(q)/batch_size)):
        question = np.array(q[i*batch_size:(i+1)*batch_size])
        temp = []
        for j in range(i*batch_size,(i+1)*batch_size):
            temp.append(valid_featuremapping[valid_image_name[j]])
        feature = Variable(torch.from_numpy(f[temp]).float()).cuda()
        target = np.array(t[i*batch_size:(i+1)*batch_size])
        # print (q,question)
        output = model(Variable(torch.from_numpy(question)).cuda(),feature,
            Variable(torch.from_numpy(target).float()).cuda())
        output = output.data.cpu().numpy()
        if len(predict) == 0:
            predict = output
        else:
            predict = np.concatenate((np.array(predict),output),axis=0)
    # print ((len(f)/batch_size)*batch_size)
    question = np.array(q[int(len(q)/batch_size)*batch_size:])
    temp = []
    for j in range(int(len(q)/batch_size)*batch_size,len(q)):
        temp.append(valid_featuremapping[valid_image_name[j]])
    feature = Variable(torch.from_numpy(f[temp]).float()).cuda()
    # feature = f[i*batch_size,:]
    target = np.array(t[int(len(q)/batch_size)*batch_size:])
    output = model(Variable(torch.from_numpy(question)).cuda(),feature,
            Variable(torch.from_numpy(target).float()).cuda())
    output = output.data.cpu().numpy()
    predict = np.concatenate((predict,output),0)
    predict = np.argmax(predict,-1)
    answer = np.argmax(a,-1)
    correct = 0
    for i in range(len(predict)):
        if predict[i] == answer[i]:
            correct += 1
    print ('testing acc',float(correct)/float(len(predict)))

def create_new_data():
    train_feature = h5py.File('/tmp2/train_nas_h5/train.hdf5','r')
    temp_feature = h5py.File('/tmp2/train_nas_h5/trainv2.hdf5','w')
    feature_index = [str(i) for i in range(1,39)]
    feature = []
    for i in feature_index:
        print ('loading:',i)
        if i == '1':
            feature = [train_feature[i][:]]
        else:
            # feature = np.concatenate((feature,train_feature[i][:]),axis=0) # (1500_ith, 50, 4036)
            feature.append(train_feature[i][:])
    feature = np.reshape(np.array(feature),(-1,50,4032))
    temp_feature.create_dataset('0',data=feature)
    feature_index = [str(i) for i in range(39,76)]
    feature = []
    for i in feature_index:
        print ('loading:',i)
        if i == '39':
            feature = [train_feature[i][:]]
        else:
            # feature = np.concatenate((feature,train_feature[i][:]),axis=0) # (1500_ith, 50, 4036)
            feature.append(train_feature[i][:])
    feature = np.reshape(np.array(feature),(-1,50,4032))
    temp_feature.create_dataset('1',data=feature)
    feature = train_feature['76'][:]
    temp_feature.create_dataset('2',data=feature)
    temp_feature.close()
    sys.exit(0)

def load_train_data():
    train_feature = h5py.File('/tmp2/train_nas_h5/trainv2.hdf5','r')
    feature_index = [str(i) for i in range(0,3)]
    feature = []
    for i in feature_index:
        print ('loading:',i)
        if i == '0':
            feature = [train_feature[i][:]]
        else:
            feature.append(train_feature[i][:])
    train_feature.close()
    return feature

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    torch.manual_seed(1000)
    # create_new_data()
    train_feature = load_train_data()
    # train_feature = [np.random.rand(50,50),np.random.rand(50,50),np.random.rand(50,50)]
    train_feature_to_question_index, train_image_idx, train_q, train_a, train_target, train_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl')

    #train 113221

    model = Model(vocab_size=len(word_embedding),
        emb_dim=300,
        feature_dim=4032,
        hidden_dim=500,
        out_dim=2,
        pretrained_embedding=word_embedding
        ).cuda()
    print ('model size',count_parameters(model))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    BATCH_SIZE = 64
    for epoch in range(args.epochs):
        loss_record = []
        start_idx = 0
        end_idx = 0
        data_count = 0
        print (len(train_feature))
        for feature_slice in train_feature:
            """
            start_idx end_idx (question index)
            """
            start_idx = end_idx
            end_idx = data_count + len(feature_slice)
            end_idx = train_feature_to_question_index[end_idx-1]
            r = torch.from_numpy(np.array([j for j in range(start_idx,end_idx)]))
            torch_dataset = Data.TensorDataset(data_tensor=r,target_tensor=r)
            loader = Data.DataLoader(dataset=torch_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
                )
            for step, (x_index,_) in enumerate(loader):
                x_index = x_index.numpy()
                # print (x_index)
                q = Variable(torch.from_numpy(train_q[x_index])).cuda()
                a = np.argmax(train_a[x_index],axis=-1)
                a = Variable(torch.from_numpy(a)).cuda()
                target = Variable(torch.from_numpy(train_target[x_index]).float()).cuda()
                temp = []
                for idx in x_index:
                    temp.append(train_image_idx[idx]-data_count)
                feature = Variable(torch.from_numpy(feature_slice[temp]).float()).cuda()

                output = model(q,feature,target)
                loss = loss_function(output, a)
                loss_record.append(loss.data[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 500 == 0 and step > 0:
                    print (step,sum(loss_record)/len(loss_record))
                    loss_record = []
            data_count += len(feature_slice)
        if epoch % 5 == 0 and epoch > 0:
            train_feature[0] = []
            validation(model)
            temp_file = h5py.File('/tmp2/train_nas_h5/trainv2.hdf5','r')
            train_feature[0] = temp_file['0'][:]
            temp_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epochs', metavar='', type=int, default=30, help='number of epochs.')
    args, unparsed = parser.parse_known_args()
    train(args)
