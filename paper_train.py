from collections import defaultdict
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from chu_liu_edmonds import decode_mst
import numpy as np
from random import sample, uniform
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy

# Based on the paper of Eliyahu Kiperwasser and Yoav Goldberg:  https://arxiv.org/pdf/1603.04351.pdf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# the Pre_Process class get both train paths (if there are more than 1 train file) and test path.
# the class process and build both train sentences and test sentences

class Pre_Process():
    def __init__(self, train_path, test_path, comp=False):
        super(Pre_Process, self).__init__()
        self.word_idx = 2
        self.pos_idx = 2
        self.sentences = []
        self.test_sentences = []
        self.get_vocabs(train_path)
        self.dim = len(self.pos_dict) + len(self.word_dict)
        self.get_train_sentences(train_path)
        self.get_test_sentences(test_path,comp)

    def sequence_indices(self, sequence, dict):
        indices = [dict[x] for x in sequence]
        return torch.tensor(indices, dtype=torch.long)

    def get_vocabs(self, train_paths):
        self.word_dict = {'ROOT':torch.tensor(0), '<unk>':torch.tensor(1)}
        self.pos_dict = {'ROOT':torch.tensor(0), '<unk>':torch.tensor(1)}
        self.word_frequencies = {}

        for file_path in train_paths:
            with open(file_path) as f:
                for line in f:
                    if (line == '\n'):
                        continue
                    splited_line = line.split('\t')
                    word, pos_tag = splited_line[1],splited_line[3]
                    if word not in self.word_dict.keys():
                        self.word_dict[word] = torch.tensor(self.word_idx)
                        self.word_frequencies[self.word_idx]= 1
                        self.word_idx += 1
                    else:
                        self.word_frequencies[int(self.word_dict[word])] += 1
                    if pos_tag not in self.pos_dict.keys():
                        self.pos_dict[pos_tag] = torch.tensor(self.pos_idx)
                        self.pos_idx += 1

    def get_train_sentences(self, train_paths):
        curr_words = ['ROOT']
        curr_tags = ['ROOT']
        curr_graph = []
        for file_path in train_paths:
            with open(file_path) as f:
                for line in f:
                    if (line == '\n'):
                        word_idx_tensor = self.sequence_indices(curr_words, self.word_dict)
                        pos_idx_tensor = self.sequence_indices(curr_tags, self.pos_dict)
                        true_tree = np.full(len(curr_words), -1)
                        for head , modifier in curr_graph:
                            true_tree[modifier] = head
                        self.sentences.append((word_idx_tensor, pos_idx_tensor, true_tree))
                        curr_words = ['ROOT']
                        curr_tags = ['ROOT']
                        curr_graph = []
                        continue
                    splited_line = line.split('\t')

                    word, pos_tag = splited_line[1], splited_line[3]
                    modifier, head = int(splited_line[0]), int(splited_line[6])
                    curr_words.append(word)
                    curr_tags.append(pos_tag)
                    curr_graph.append((head, modifier))
        self.num_sentences = len(self.sentences)

    def get_test_sentences(self, test_path, comp):
        curr_words = ['ROOT']
        curr_tags = ['ROOT']
        curr_graph = []
        with open(test_path) as f:
            for line in f:
                if (line == '\n'):
                    word_idx_tensor = self.sequence_indices(curr_words, self.word_dict)
                    pos_idx_tensor = self.sequence_indices(curr_tags, self.pos_dict)
                    true_tree = np.full(len(curr_words), -1)
                    if not comp:
                        for head, modifier in curr_graph:
                            true_tree[modifier] = head
                    self.test_sentences.append((word_idx_tensor, pos_idx_tensor, true_tree))
                    curr_words = ['ROOT']
                    curr_tags = ['ROOT']
                    curr_graph = []
                    continue
                splited_line = line.split('\t')

                word, pos_tag = splited_line[1], splited_line[3]
                if word not in self.word_dict.keys():
                    word = '<unk>'
                if pos_tag not in self.pos_dict.keys():
                    pos_tag = '<unk>'
                if not comp:
                    modifier, head = int(splited_line[0]), int(splited_line[6])
                else:
                    modifier, head = (splited_line[0]), (splited_line[6])
                curr_graph.append((head, modifier))
                curr_words.append(word)
                curr_tags.append(pos_tag)


    def __len__(self):
        return self.num_sentences

    def __getitem__(self, index):
        return self.sentences[index]


def NLLLOSS(score_matrix,true_tree_heads):
    sentence_len = len(true_tree_heads)
    score_matrix = torch.exp(score_matrix)
    res = torch.tensor(0).to(device)
    for modifier,head in enumerate(true_tree_heads):
        nominator = score_matrix[head][modifier]
        denominator = score_matrix.sum(0)[modifier]
        res = torch.add(res,torch.log(nominator/denominator))
    a = torch.tensor(-1/len(true_tree_heads))  # the size of Yi is the number of edges of the true graph
    return torch.mul(res,a)


class KiperwasserDependencyParser(nn.Module):
    def __init__(self,word_emb_dim,pos_emb_dim, lstm_hidden_dim,mlp_hidden_dim,num_layers, word_voca_len,pos_voca_len):
        super(KiperwasserDependencyParser, self).__init__()
        self.word_embedding = nn.Embedding(word_voca_len, word_emb_dim)
        self.pos_embedding = nn.Embedding(pos_voca_len, pos_emb_dim)
        self.hidden_dim = word_emb_dim + pos_emb_dim
        self.encoder = nn.LSTM(self.hidden_dim, lstm_hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.edge_scorer = nn.Sequential(
            nn.Linear(lstm_hidden_dim*4, mlp_hidden_dim),  #the input is 2 vectors with dim=lstm_hidden_dim concatenated
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        # self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function = NLLLOSS

    def forward(self, sentence, predict=False):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        sentence_len = len(word_idx_tensor)
        words_embeds = self.word_embedding(word_idx_tensor.to(device)).to(device)   # [sentence_length, WORD_EMB_DIM] TODO put them in cuda
        pos_embeds = self.pos_embedding(pos_idx_tensor.to(device)).to(device)         # [sentence_length, POS_EMB_DIM]

        concat = torch.cat((words_embeds,pos_embeds),1)

        concat = concat.unsqueeze(0) # [1, sentence_length, POS_EMB_DIM+WORD_EMB_DIM]
        lstm_out, _ = self.encoder(concat)  # [1, sentence_length, LSTM_HIDDEN_DIM*2]

        score_matrix = self.edge_scorer(torch.cat([lstm_out.view(lstm_out.shape[1],lstm_out.shape[2]).unsqueeze(1).repeat(1,sentence_len,1),lstm_out.repeat(sentence_len,1,1)], -1)).view(sentence_len,sentence_len)


        loss = self.loss_function(score_matrix, true_tree_heads)
        if predict:
            predicted_tree, _ = self.decoder(score_matrix.cpu().detach().numpy(),sentence_len, has_labels=False)
            # Calculate the negative log likelihood loss described above
            return loss, predicted_tree
        else:
            return loss


if __name__ == '__main__':
    train_path = "train.labeled"
    test_path = "test.labeled"
    data = Pre_Process([train_path],test_path)
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    WORD_EMB_DIM = 100
    POS_EMB_DIM = 25
    LSTM_HIDDEN_DIM = 125
    MLP_HIDDEN_DIM = 100

    INPUT_SIZE = len(data.word_dict) + len(data.pos_dict)
    NUM_LAYERS = 2
    LEARNING_RATE = 0.01
    BATCH_SIZE = 50
    EPOCHS = 10


    model = KiperwasserDependencyParser(WORD_EMB_DIM
                                        ,POS_EMB_DIM,LSTM_HIDDEN_DIM,MLP_HIDDEN_DIM,NUM_LAYERS,len(data.word_dict),
                                        len(data.pos_dict)).to(device)
    # model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)




    params = {}
    correct_edges = 0
    total_edges = 0
    model.zero_grad()
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(EPOCHS):
        print(f"start epoch {epoch}")
        i = 1
        printable_loss = 0
        shuffeled_sentences = sample(data.sentences, len(data.sentences))

        for sentence in shuffeled_sentences:
            for idx in sentence[0][1:]:
                p = uniform(0, 1)
                if p < 0.25 / (0.25 +data.word_frequencies[int(idx)]):
                    idx = torch.tensor(1) # the index of <unk>
            loss, predicted_tree = model(sentence, predict=True)
            loss = loss/BATCH_SIZE
            loss.backward()
            _,_, true_tree = sentence
            if i % BATCH_SIZE == 0:
                optimizer.step()
                model.zero_grad()
            printable_loss += loss.item()
            for j in range(len(true_tree)):
                if true_tree[j] == predicted_tree[j]:
                    correct_edges += 1
                total_edges += 1
            i+=1
        train_acc = correct_edges/total_edges
        print(f"current UAS for train for epoch number {epoch} : {train_acc}")
        printable_loss = BATCH_SIZE * (printable_loss / len(shuffeled_sentences))
        train_loss_list.append(float(printable_loss))
        train_accuracy_list.append(float(train_acc))

        # PREDICTION

        printable_loss = 0
        correct_edges = 0
        total_edges = 0
        for sentence in data.test_sentences:
            loss, predicted_tree = model(sentence, predict=True)
            _, _, true_tree = sentence
            printable_loss += loss.item()
            for i in range(len(true_tree)):
                if true_tree[i] == predicted_tree[i]:
                    correct_edges += 1
                total_edges += 1
        printable_loss = printable_loss / len(data.test_sentences)
        test_loss_list.append(float(printable_loss))
        test_acc = correct_edges/total_edges
        print(f" UAS for test  : {test_acc}")
        test_accuracy_list.append(float(test_acc))

        # for param_tensor in model.state_dict():
        #     if epoch==0:
        #         params[param_tensor] = deepcopy(model.state_dict()[param_tensor])
        #     else:
        #         if torch.all(torch.eq(params[param_tensor], model.state_dict()[param_tensor])):
        #             print(f"{param_tensor} didnt changed in epoch {epoch}")
        #         params[param_tensor] = deepcopy(model.state_dict()[param_tensor])
        correct_edges = 0
        total_edges = 0

    import matplotlib.pyplot as plt

    plt.plot(train_accuracy_list, c="blue", label="train UAS Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    # plt.savefig('basic train accuracy-epochs.png')
    plt.plot(test_accuracy_list, c="red", label="test UAS Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('basic accuracy-epochs.png')

    plt.clf()

    plt.plot(train_loss_list, c="blue", label="train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    # plt.savefig('basic train loss-epochs.png')

    plt.plot(test_loss_list, c="red", label="test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('basic loss-epochs.png')

    torch.save(model.state_dict() , "basic_model.pickle")
    print("done!!!!")

