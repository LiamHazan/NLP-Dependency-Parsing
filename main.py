from train import Pre_Process, FreeDependencyParser
from torchtext.vocab import Vocab
import torchtext
import numpy as np
import torch.optim as optim
import torch
from random import sample


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = "train.labeled"
    test_path = "test.labeled"
    data = Pre_Process([train_path], test_path)

    LEARNING_RATE = 0.001
    EPOCHS = 60
    BATCH_SIZEs = [10]
    WORD_EMB_DIMs = [100]
    PRE_WORD_EMB_DIMs = [100]
    POS_EMB_DIMs = [100]
    MLP_HIDDEN_DIMs = [300]
    NUM_LAYERSs = [3]
    LSTM_HIDDEN_DIMs = [250]
    p_dropout = 0.25





    BATCH_SIZE = 1
    WORD_EMB_DIM = 100
    PRE_WORD_EMB_DIM = 300
    POS_EMB_DIM = 25
    LSTM_HIDDEN_DIM = 125
    MLP_HIDDEN_DIM = 100

    INPUT_SIZE = len(data.word_dict) + len(data.pos_dict)
    NUM_LAYERS = 2
    results = {}

    for BATCH_SIZE in BATCH_SIZEs:
        for WORD_EMB_DIM in WORD_EMB_DIMs:
            for PRE_WORD_EMB_DIM in PRE_WORD_EMB_DIMs:
                for POS_EMB_DIM in POS_EMB_DIMs:
                    for MLP_HIDDEN_DIM in MLP_HIDDEN_DIMs:
                        for LSTM_HIDDEN_DIM in LSTM_HIDDEN_DIMs:
                            for NUM_LAYERS in NUM_LAYERSs:

                                glove = torchtext.vocab.GloVe(name="6B", dim=PRE_WORD_EMB_DIM)
                                # prepare embedding matrix
                                embedding_matrix = np.zeros((len(data.word_dict), PRE_WORD_EMB_DIM))
                                for word, i in data.word_dict.items():
                                    embedding_vector = glove[word]
                                    if embedding_vector is not None:
                                        # words not found in embedding index will be all-zeros.
                                        embedding_matrix[i] = embedding_vector
                                embedding_matrix = torch.FloatTensor(embedding_matrix).to(device)

                                model = FreeDependencyParser(embedding_matrix, PRE_WORD_EMB_DIM, WORD_EMB_DIM
                                                             , POS_EMB_DIM, LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM,
                                                             NUM_LAYERS, len(data.word_dict),
                                                             len(data.pos_dict), p_dropout).to(device)

                                optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9), lr=LEARNING_RATE,
                                                       weight_decay=1e-5)
                                params = {}
                                correct_edges = 0
                                total_edges = 0
                                model.zero_grad()
                                max_res = 0
                                max_params = ""
                                train_accuracy_list = []
                                test_accuracy_list = []
                                train_loss_list = []
                                test_loss_list = []
                                for epoch in range(EPOCHS):
                                    print(f"start epoch {epoch}")
                                    i = 1
                                    shuffeled_sentences = sample(data.sentences, len(data.sentences))
                                    printable_loss = 0
                                    for sentence in shuffeled_sentences:
                                        loss, predicted_tree = model(sentence, predict=True)
                                        loss = loss / BATCH_SIZE
                                        loss.backward()
                                        _, _, true_tree = sentence
                                        if i % BATCH_SIZE == 0:
                                            optimizer.step()
                                            # for p in not_changed:
                                            #     print(model.state_dict()[p].grad)
                                            model.zero_grad()
                                        printable_loss += loss.item()

                                        # print(f"true_tree = {true_tree}")
                                        # print(f"predicted_tree = {predicted_tree}")
                                        for j in range(len(true_tree)):
                                            if true_tree[j] == predicted_tree[j]:
                                                correct_edges += 1
                                            total_edges += 1
                                        i += 1
                                    train_accuracy_list.append(correct_edges / total_edges)
                                    printable_loss = BATCH_SIZE * (printable_loss / len(shuffeled_sentences))
                                    train_loss_list.append(float(printable_loss))
                                    print(
                                        f"current UAS for train for epoch number {epoch} : {correct_edges / total_edges}")
                                    printable_loss = 0
                                    correct_edges = 0
                                    total_edges = 0
                                    for sentence in data.test_sentences:
                                        loss, predicted_tree = model(sentence, predict=True)
                                        _, _, true_tree = sentence
                                        printable_loss += loss.item()


                                        # print(f"true_tree =      {true_tree}")
                                        # print(f"predicted_tree = {predicted_tree}")
                                        for i in range(len(true_tree)):
                                            if true_tree[i] == predicted_tree[i]:
                                                correct_edges += 1
                                            total_edges += 1

                                    printable_loss = (printable_loss / len(data.test_sentences))
                                    test_loss_list.append(printable_loss)
                                    print(f" UAS for test  : {correct_edges / total_edges}")
                                    test_accuracy_list.append(correct_edges / total_edges)

                                    if (correct_edges / total_edges) < 0.09:
                                        break
                                    if (correct_edges / total_edges) > max_res:
                                        max_res = (correct_edges / total_edges)
                                        max_params = f"batch size= {BATCH_SIZE}, word emb dim= {WORD_EMB_DIM}, pre word emb dim= {PRE_WORD_EMB_DIM}" \
                                                     f" pos dim= {POS_EMB_DIM}, mlp hid dim= {MLP_HIDDEN_DIM}, num layers= {NUM_LAYERS}, epoch= {epoch}"
                                    correct_edges = 0
                                    total_edges = 0
                                results[max_params] = max_res

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
    plt.savefig('free accuracy-epochs.png')

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
    plt.savefig('free loss-epochs.png')

    print(results)
