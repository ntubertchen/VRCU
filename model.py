import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from indrnn import IndRNN
import numpy as np

from transformer import make_tf

class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, feature_dim, hidden_dim,
        out_dim, pretrained_embedding, args):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # indrnn.IndRNN()
        # def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False,
        #          batch_first=False, bidirectional=False, **kwargs):
        self.lstm = nn.LSTM(emb_dim, hidden_dim,bidirectional=True)
        # self.lstm = indrnn.IndRNN(emb_dim, hidden_dim)
        self.image_lstm_x = nn.LSTM(99+hidden_dim *2, hidden_dim,bidirectional=True)
        self.image_lstm_y = nn.LSTM(99+hidden_dim *2, hidden_dim,bidirectional=True)
        # self.image_lstm = indrnn.IndRNN(99, hidden_dim)
        # weight
        self.gt_W_image_attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.gt_W_prime_image_attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.gt_W_question = nn.Linear(hidden_dim, hidden_dim)
        self.gt_W_prime_question = nn.Linear(hidden_dim, hidden_dim)
        self.gt_W_image = nn.Linear(feature_dim, hidden_dim)
        self.gt_W_prime_image = nn.Linear(feature_dim, hidden_dim)
        self.gt_W_image_lstm = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.gt_W_prime_image_lstm = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.gt_W_whole_image = nn.Linear(hidden_dim+99, hidden_dim)
        self.gt_W_prime_whole_image = nn.Linear(hidden_dim+99, hidden_dim)

        self.gt_W_clf = nn.Linear(hidden_dim+99, hidden_dim+99)
        self.gt_W_prime_clf = nn.Linear(hidden_dim+99, hidden_dim+99)

        self.word_embedding = nn.Embedding(vocab_size, emb_dim)

        self.image_word_attention = nn.Linear(hidden_dim, 1)
        self.whole_image_and_word_attention = nn.Linear(hidden_dim,1)
        self.clf = nn.Linear(hidden_dim * 4, out_dim)

        # assign pretrained embedding
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
    
    def lstm_image(self, question, image_feature_x, image_feature_y, target):
        """
        question (batch, 20) -> I think it should be (batch, q_len) ?
        image_feature (batch, 50, 4036) -> I think it should be (batch, 20, 4036) ?
        gt (batch,20,98) -> what is this?
        target (batch, 98)
        """

        #get question hidden state
        embedd_question = self.word_embedding(question)
        lstm_out, (h,c) = self.lstm(embedd_question.permute(1, 0, 2))
        lstm_hiddenstate = c #(layer * dir, B, hidden) == (2, B, hidden)
        lstm_hiddenstate = lstm_hiddenstate.permute(1, 0, 2)
        lstm_hiddenstate_reshape = lstm_hiddenstate.contiguous().view(-1, self.hidden_dim*2)
        language_prior = lstm_hiddenstate_reshape # (B, 2*hidden)
        
        #x_axis lstm with language prior
        language_prior = torch.unsqueeze(language_prior,1) # (batch, 1, hidden * 2)
        language_prior_reshape = language_prior.repeat(1, 20, 1) # (batch, 20, hidden * 2)
        language_prior = torch.squeeze(language_prior)
        
        image_x_with_prior = torch.cat((image_feature_x,language_prior_reshape),-1)
        image_y_with_prior = torch.cat((image_feature_y,language_prior_reshape),-1)
        
        _, (x_h, x_c) = self.image_lstm_x(image_x_with_prior.permute(1, 0, 2))
        image_x_lstm = x_c.permute(1,0,2).contiguous().view(-1,self.hidden_dim*2)

        _, (y_h, y_c) = self.image_lstm_y(image_y_with_prior.permute(1, 0, 2))
        image_y_lstm = y_c.permute(1,0,2).contiguous().view(-1,self.hidden_dim*2)
        activated_image_x_lstm = self._gated_tanh(image_x_lstm, self.gt_W_image_lstm, self.gt_W_prime_image_lstm) #(batch, hidden*2)
        activated_image_y_lstm = self._gated_tanh(image_y_lstm, self.gt_W_image_lstm, self.gt_W_prime_image_lstm)
        prediction = self.clf(torch.cat((activated_image_x_lstm,activated_image_y_lstm),-1))

        return prediction

    def simple(self, question, image_feature, target):
        """
        question (batch, 20)
        image_feature (batch, 50, 4036)
        gt (batch,20,98)
        target (batch, 98)
        """

        #get question hidden state
        embedd_question = self.word_embedding(question)
        lstm_out, (h,c) = self.lstm(embedd_question.permute(1, 0, 2))
        # lstm_hiddenstate = torch.cat((h,c),-1)
        lstm_hiddenstate = h
        language_prior = lstm_hiddenstate[-1]
        given_target = torch.cat((language_prior,target),-1)

        prediction = self.clf(self._relu(given_target, self.gt_W_clf, self.gt_W_prime_clf))

        return prediction

    def forward(self, question, image_feature, target):
        """
        question (batch, 20)
        image_feature (batch, 50, 4036)
        target (batch, 98)
        """

        #get question hidden state
        embedd_question = self.word_embedding(question)
        lstm_out, (h,c) = self.lstm(embedd_question.permute(1, 0, 2))
        # lstm_hiddenstate = torch.cat((h,c),-1)
        lstm_hiddenstate = h
        language_prior = lstm_hiddenstate[-1]
        # image_feature = F.normalize(image_feature, -1) # (batch, 50, 4032) cancel for experiment

        #image attention 
        language_prior = torch.unsqueeze(language_prior,1) # (batch, 1, hidden)
        language_prior_reshape = language_prior.repeat(1, 20, 1) # (batch, 50, hidden)
        language_prior = torch.squeeze(language_prior)
        # print (image_feature,language_prior_reshape)
        image_and_word = torch.cat((image_feature, language_prior_reshape), -1)
        # print (image_and_word)
        image_and_word = self._gated_tanh(image_and_word, self.gt_W_image_attention, self.gt_W_prime_image_attention)

        alpha = self.image_word_attention(image_and_word)
        alpha = F.softmax(alpha.squeeze())

        # print (alpha.squeeze(),alpha.unsqueeze(1), image_feature)
        attended_image = torch.bmm(alpha.unsqueeze(1), image_feature).squeeze() # (batch, 4032)

        # element wise dot product
        activated_language_prior = self._gated_tanh(language_prior, self.gt_W_question, self.gt_W_prime_question)
        activated_image_feature = self._gated_tanh(attended_image, self.gt_W_image, self.gt_W_prime_image)
        masked_image_feature = torch.mul(activated_language_prior, activated_image_feature)

        given_target = torch.cat((masked_image_feature,target),-1)
        prediction = self.clf(self._gated_tanh(given_target, self.gt_W_clf, self.gt_W_prime_clf))

        return prediction

    def GLU(self,x,W,W_prime):
        y_tilde = W(x)
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y

    def _gated_tanh(self, x, W, W_prime):
        #x = F.dropout(x, p=0.5, training=self.training)
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y

    def _relu(self,x,W,W_prime):
        return F.relu(W(x))