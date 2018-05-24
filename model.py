import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, feature_dim, hidden_dim,
        out_dim, pretrained_embedding):
        super(Model, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(emb_dim, hidden_dim)

        # weight
        self.gt_W_image_attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.gt_W_prime_image_attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.gt_W_question = nn.Linear(hidden_dim, hidden_dim)
        self.gt_W_prime_question = nn.Linear(hidden_dim, hidden_dim)
        self.gt_W_image = nn.Linear(feature_dim, hidden_dim)
        self.gt_W_prime_image = nn.Linear(feature_dim, hidden_dim)
        self.gt_W_clf = nn.Linear(hidden_dim+94, hidden_dim+94)
        self.gt_W_prime_clf = nn.Linear(hidden_dim+94, hidden_dim+94)

        self.word_embedding = nn.Embedding(vocab_size, emb_dim)

        self.image_word_attention = nn.Linear(hidden_dim, 1)
        self.clf = nn.Linear(hidden_dim+94, out_dim)

        # assign pretrained embedding
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
    
    def forward(self, question, image_feature, target):
        """
        question (batch, 20)
        image_feature (batch, 50, 4036)
        """

        #get question hidden state
        embedd_question = self.word_embedding(question)
        lstm_out, lstm_hiddenstate = self.lstm(embedd_question.permute(1, 0, 2))
        language_prior = lstm_hiddenstate[-1]
        image_feature = F.normalize(image_feature, -1) # (batch, 50, 4036)

        #image attention 
        language_prior_reshape = language_prior.repeat(1, 50).view(-1, 50, self.hidden_dim)
        image_and_word = torch.cat((image_feature, language_prior_reshape), -1)
        image_and_word = self._gated_tanh(image_and_word, self.gt_W_image_attention, self.gt_W_prime_image_attention)

        alpha = self.image_word_attention(image_and_word)
        alpha = F.softmax(a.squeeze())
        attended_image = torch.bmm(a.unsqueeze(1), image_feature).squeeze() # (batch, 4036)

        # element wise dot product
        activated_language_prior = self._gated_tanh(language_prior, self.gt_W_question, self.gt_W_prime_question)
        activated_image_feature = self._gated_tanh(attended_image, self.gt_W_image, self.gt_W_prime_image)
        masked_image_feature = torch.mul(activated_language_prior, activated_image_feature)

        given_target = torch.cat((masked_image_feature,target),-1)

        prediction = self.clf(self._gated_tanh(masked_image_feature, self.gt_W_clf. self.gt_W_prime_clf))

        return prediction

    def _gated_tanh(self, x, W, W_prime):
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y