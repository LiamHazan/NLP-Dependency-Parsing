from paper_train import Pre_Process, KiperwasserDependencyParser
import torch
from random import sample, uniform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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