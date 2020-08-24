from paper_train import Pre_Process, NLLLOSS, KiperwasserDependencyParser
from train import FreeDependencyParser
import torch
import torchtext
import numpy as np
from random import sample
import torch.optim as optim

# this py file train both models and save their trained parameters in pickle files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = "train.labeled"
test_path = "test.labeled"
comp_path = "comp.unlabeled"
data = Pre_Process([train_path, test_path], comp_path, comp=True)

WORD_EMB_DIM = 100
POS_EMB_DIM = 25
LSTM_HIDDEN_DIM = 125
MLP_HIDDEN_DIM = 100

INPUT_SIZE = len(data.word_dict) + len(data.pos_dict)
NUM_LAYERS = 2
LEARNING_RATE = 0.01
BATCH_SIZE = 50
EPOCHS = 15



basic_model = KiperwasserDependencyParser(WORD_EMB_DIM
                                        ,POS_EMB_DIM,LSTM_HIDDEN_DIM,MLP_HIDDEN_DIM,NUM_LAYERS,len(data.word_dict),
                                        len(data.pos_dict)).to(device)



optimizer = torch.optim.Adam(basic_model.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    print(f"start epoch {epoch}")
    shuffeled_sentences = sample(data.sentences, len(data.sentences))
    i=0
    for sentence in shuffeled_sentences:
        loss, predicted_tree = basic_model(sentence, predict=True)
        loss = loss / BATCH_SIZE
        loss.backward()
        _, _, true_tree = sentence
        if i % BATCH_SIZE == 0:
            optimizer.step()
            basic_model.zero_grad()
        i += 1



torch.save(basic_model.state_dict() , "basic_model.pickle")



# FREE MODEL
LEARNING_RATE = 0.001
BATCH_SIZE = 10
WORD_EMB_DIM = 100
PRE_WORD_EMB_DIM = 100
POS_EMB_DIM = 100
LSTM_HIDDEN_DIM = 200
MLP_HIDDEN_DIM = 300
p_dropout = 0.25
INPUT_SIZE = len(data.word_dict) + len(data.pos_dict)
NUM_LAYERS = 3
EPOCHS = 50

glove = torchtext.vocab.GloVe(name="6B", dim=PRE_WORD_EMB_DIM)
# prepare embedding matrix
embedding_matrix = np.zeros((len(data.word_dict), PRE_WORD_EMB_DIM))
for word, i in data.word_dict.items():
    embedding_vector = glove[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
embedding_matrix = torch.FloatTensor(embedding_matrix).to(device)

free_model = FreeDependencyParser(embedding_matrix, PRE_WORD_EMB_DIM, WORD_EMB_DIM
                                                                    , POS_EMB_DIM, LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM,
                                                                    NUM_LAYERS, len(data.word_dict),
                                                                    len(data.pos_dict),p_dropout).to(device)
optimizer = optim.Adam(free_model.parameters(), betas=(0.9, 0.9), lr=LEARNING_RATE, weight_decay=1e-5)

for epoch in range(EPOCHS):
    print(f"start epoch {epoch}")
    shuffeled_sentences = sample(data.sentences, len(data.sentences))
    i=0
    correct_edges = 0
    total_edges = 0
    for sentence in shuffeled_sentences:
        loss, predicted_tree = free_model(sentence, predict=True)
        loss = loss / BATCH_SIZE
        loss.backward()
        _, _, true_tree = sentence
        if i % BATCH_SIZE == 0:
            optimizer.step()
            free_model.zero_grad()
        for j in range(len(true_tree)):
            if true_tree[j] == predicted_tree[j]:
                correct_edges += 1
            total_edges += 1
        i += 1
    print(f"current UAS for train for epoch number {epoch} : {correct_edges / total_edges}")
    if epoch % 20 == 0:
        torch.save(free_model.state_dict(), "free_model.pickle")

torch.save(free_model.state_dict() , "free_model.pickle")
