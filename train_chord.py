import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from network import *
from until import *


BATCH_SIZE = 128
EMBEDDING_SIZE = 80
HIDDENG_SIZE = 100
NLAYERS = 10
GRAD_CLIP = 1
NUM_EPOCH = 50
USE_CUDA = torch.cuda.is_available()
DEVICE = ("cuda" if USE_CUDA else "cpu")


def train_chord():
    
    train_input, train_output, num_chords = prepare_train_chord()
    print(num_chords)
    val_input, val_output = prepare_val_chord()
    model = LSTM_BiDir(num_chords, EMBEDDING_SIZE, HIDDENG_SIZE, NLAYERS, dropout=0.5)
    if USE_CUDA:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    val_losses = []
    # if os.path.exists("model/chord_best.pth"):
    #     model.load_state_dict(torch.load("model/chord_best.pth"))
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    
    train_loss_list = []
    val_loss_list = []
    countx = 0
    fail_count = 0
    for epoch in range(NUM_EPOCH):
        model.train()
        hidden = model.init_hidden(BATCH_SIZE)
        
        for start in range(0, len(train_input) - BATCH_SIZE, BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = torch.LongTensor(train_input[start:end])
            batchY = torch.LongTensor(train_output[start:end])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 10), batchY.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            print("Epoch :", epoch, loss.item())
            if not countx % 10:
                train_loss_list.append(loss.item())
                val_loss = evaluate(model, val_input, val_output, loss_fn, num_chords)
                val_loss_list.append(val_loss)
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    val_losses.append(val_loss)
                    print("best_model, val_loss :", val_loss)
                    # torch.save(model.state_dict(), "model/chord_best.pth")
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count == 3:
                        scheduler.step()
                        # optimizer = torch.optim.SGD(model.parameters(), learning_rate)
                        fail_count = 2
                # if learning_rate < 0.1:
                #     model.load_state_dict(torch.load("model/chord_best.pth"))
                #     optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
                #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
            countx += 1
    draw(train_loss_list, val_loss_list)
                

def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden)


def evaluate(model, val_input, val_output, loss_fn, num_chords):
    model.eval()
    total_loss = 0.
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for start in range(0, len(val_input) - BATCH_SIZE, BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = torch.LongTensor(val_input[start:end])
            batchY = torch.LongTensor(val_output[start:end])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 10), batchY.view(-1))
            total_count += float(len(val_input) * len(val_output)) * 100.
            total_loss += float(loss.item()) * float(len(val_input) * 10. * float(len(val_output))) * 10.
    loss = total_loss / total_count
    model.train()
    return loss


def draw(train_loss_list, val_loss_list):
    x1 = range(0, len(train_loss_list))
    y1 = train_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title("chord_loss(Adam),Embedding-size=80")
    plt.ylabel("train_loss")
    x2 = range(0, len(val_loss_list))
    y2 = val_loss_list  
    plt.subplot(2,1,2)
    plt.plot(x2, y2, 'b--')
    plt.xlabel("BATCH_ID")
    plt.ylabel("val_loss")
    plt.savefig("model/chord_loss(Adam)80.jpg")
    plt.show()


if __name__ == '__main__':
    train_chord()
