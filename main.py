import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as np
import h5py
import argparse

from model import Model 
import utils

def valid(args):
    torch.manual_seed(1000)

    #train_featuremapping, train_image_name, train_q, train_a, train_target, train_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl')
    valid_featuremapping, valid_image_name, valid_q, valid_a, valid_target, valid_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl')
    valid_feature = h5py.File('/tmp2/val_nas_h5/valid.hdf5')
    #train 113221

    model = Model(vocab_size=len(word_embedding),emb_dim=300,feature_dim=4032,hidden_dim=500,out_dim=3,pretrained_embedding=word_embedding,).cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    BATCH_SIZE = 4
    print ('valid len',len(valid_feature['train']))
    for epoch in range(args.epochs):
        loss_record = []
        r = torch.from_numpy(np.array([i for i in range(len(valid_feature['train']))]))
        torch_dataset = Data.TensorDataset(data_tensor=r,target_tensor=r)
        loader = Data.DataLoader(dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
            )
        feature_map = valid_feature['train'][:]
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
            print (loss)
            loss_record.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train(args):
    torch.manual_seed(1000)

    train_featuremapping, train_image_name, train_q, train_a, train_target, train_arxiv, word_embedding = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.train.new.jsonl')
    #valid_featuremapping, valid_q, valid_a, valid_target, valid_arxiv, _ = utils.load_data('/home/alas79923/vqa/faster-rcnn.pytorch/guesswhat.valid.new.jsonl')
    train_feature = h5py.File('/tmp2/train_nas_h5/train.hdf5')
    #train 113221

    model = Model(vocab_size=len(word_embedding),
        emb_dim=300,
        feature_dim=4036,
        hidden_dim=500,
        out_dim=3,
        pretrained_embedding=word_embedding
        ).cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    BATCH_SIZE = 15
    for epoch in range(args.epochs):
        loss_record = []
        for i in range((113221/1500)+1):
            if i == 113221/1500:
                r = torch.from_numpy(np.array([j for j in range(i*1500,113221)]))
            else:
                r = torch.from_numpy(np.array([j for j in range(i*1500,(i+1)*1500)]))
            torch_dataset = Data.TensorDataset(data_tensor=r,target_tensor=r)
            loader = Data.DataLoader(dataset=torch_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
                )
            feature_map = train_feature[str(i)][:]
            for step, (x_index,_) in enumerate(loader):
                q = Variable(torch.from_numpy(train_q[x_index])).cuda()
                a = np.argmax(train_a[x_index],axis=-1)
                a = Variable(torch.from_numpy(a)).cuda()
                target = Variable(torch.from_numpy(train_target[x_index]).float()).cuda()
                temp = []
                offset = i*1500
                for idx in x_index:
                    temp.append(train_featuremapping[train_image_name[idx]]-offset)
                feature = Variable(torch.from_numpy(feature_map[temp]).float()).cuda()

                output = model(q,feature,target)
                loss = loss_function(output, a)
                print (loss)
                loss_record.append(loss.data[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epochs', metavar='', type=int, default=10, help='number of epochs.')
    args, unparsed = parser.parse_known_args()
    valid(args)

