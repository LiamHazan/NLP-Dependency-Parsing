from collections import Counter, defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from chu_liu_edmonds import decode_mst
from random import  sample
from copy import deepcopy
import pickle
from paper_train import NLLLOSS, Pre_Process

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




class FreeDependencyParser(nn.Module):
    def __init__(self,embedding_matrix,pre_word_emb_dim, word_emb_dim,pos_emb_dim, lstm_hidden_dim,mlp_hidden_dim,num_layers, word_voca_len,pos_voca_len, p_dropout):
        super(FreeDependencyParser, self).__init__()
        self.word_embedding = nn.Embedding(word_voca_len, word_emb_dim)
        self.pre_trained_word_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.pos_embedding = nn.Embedding(pos_voca_len, pos_emb_dim)
        self.hidden_dim = word_emb_dim + pos_emb_dim+ pre_word_emb_dim
        self.encoder = nn.LSTM(self.hidden_dim, lstm_hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=p_dropout)
        self.edge_scorer = nn.Sequential(
            nn.Linear(lstm_hidden_dim*4, mlp_hidden_dim),  #the input is 2 vectors with dim=lstm_hidden_dim concatenated
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function = NLLLOSS

    def forward(self, sentence, predict=False):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence
        sentence_len = len(word_idx_tensor)
        pre_trained_words_embeds = self.pre_trained_word_embedding(word_idx_tensor.to(device))   # [sentence_length, WORD_EMB_DIM]
        words_embeds = self.word_embedding(word_idx_tensor.to(device)).to(device)   # [sentence_length, WORD_EMB_DIM]
        pos_embeds = self.pos_embedding(pos_idx_tensor.to(device)).to(device)         # [sentence_length, POS_EMB_DIM]

        concat = torch.cat((words_embeds,pre_trained_words_embeds,pos_embeds),1)

        concat = concat.unsqueeze(0) # [1, sentence_length, POS_EMB_DIM+WORD_EMB_DIM+PRE_WORD_EMB_DIM]
        lstm_out, _ = self.encoder(concat)  # [1, sentence_length, LSTM_HIDDEN_DIM*2]
        score_matrix = self.edge_scorer(torch.cat([lstm_out.view(lstm_out.shape[1],lstm_out.shape[2]).unsqueeze(1).repeat(1,sentence_len,1),lstm_out.repeat(sentence_len,1,1)], -1)).view(sentence_len,sentence_len)

        # for the words who are in long distance from each other we deficit their weight
        for i in range(1,score_matrix.shape[0],sentence_len):
            for j in range(1,score_matrix.shape[0],sentence_len):
                if abs(j - i) > 0.7*sentence_len:
                    score_matrix[i][j] -= 100

        loss = self.loss_function(score_matrix, true_tree_heads)
        if predict:
            predicted_tree, _ = self.decoder(score_matrix.cpu().detach().numpy(),sentence_len, has_labels=False)
            # Calculate the negative log likelihood loss described above
            return loss, predicted_tree
        else:
            return loss


