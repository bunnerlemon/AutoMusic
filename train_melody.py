import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from network import *
from until import *


BATCH_SIZE = 128
EMBEDDING_SIZE = 20
HIDDENG_SIZE = 100
NLAYERS = 10
GRAD_CLIP = 1
NUM_EPOCH = 5
USE_CUDA = torch.cuda.is_available()
DEVICE = ("cuda" if USE_CUDA else "cpu")


def train_melody():
    train_melody_input, train_melody_output = prepare_train_melody()
    val_melody_input, val_melody_output = prepare_val_melody()
    
    melodys = read_("data/melody/melodys")
    melody_names = sorted(set(melodys))
    melody2int = dict((melody, num) for num, melody in enumerate(melody_names))

    model = ThreeLayerLSTM(len(melody_names), EMBEDDING_SIZE, HIDDENG_SIZE, NLAYERS, dropout=0.3)
    if USE_CUDA:
        model = model.cuda()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    fail_count = 0
    countx = 0
    val_losses = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(NUM_EPOCH):
        model.train()
        hidden = model.init_hidden(BATCH_SIZE)
        for start_idx in range(0, len(train_melody_input) - BATCH_SIZE, BATCH_SIZE):
            end_idx = start_idx + BATCH_SIZE
            
            batchX = torch.LongTensor(train_melody_input[start_idx:end_idx])
            batchY = torch.LongTensor(train_melody_output[start_idx:end_idx])

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
            print("Epoch", epoch, loss.item())
            
            if countx % 20 == 0:
                train_loss_list.append(loss.item())
                val_loss = evaluate(model, val_melody_input, val_melody_output, loss_fn)
                val_loss_list.append(val_loss)
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    print("best_melody_loss:", val_loss)
                    torch.save(model.state_dict(), "model/melody.pth")
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count == 3:
                        scheduler.step()
                        fail_count = 2
            countx += 1
    draw(train_loss_list, val_loss_list)

def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden)


def evaluate(model, val_melody_input, val_melody_output, loss_fn):
    model.eval()
    total_count = 0.
    total_loss = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for start_idx in range(0, len(val_melody_input) - BATCH_SIZE, BATCH_SIZE):
            end_idx = start_idx + BATCH_SIZE
            batchX = torch.LongTensor(val_melody_input[start_idx:end_idx])
            batchY = torch.LongTensor(val_melody_output[start_idx:end_idx])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 10), batchY.view(-1))
            total_loss += float(loss.item()) * float(len(val_melody_input)) * 10. * float(len(val_melody_output)) * 10.
            total_count += float(len(val_melody_input)) * float(len(val_melody_output)) * 100.
    loss = total_loss / total_count
    model.train()
    return loss


def draw(train_loss_list, val_loss_list):
    x1 = range(0, len(train_loss_list))
    y1 = train_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title("Medloy_loss(SGD)")
    plt.ylabel("train_loss")
    x2 = range(0, len(val_loss_list))
    y2 = val_loss_list
    plt.subplot(2,1,2)
    plt.plot(x2, y2, 'b--')
    plt.xlabel("BATCH_ID")
    plt.ylabel("val_loss")
    plt.savefig("model/melody_loss(SGD).jpg")
    plt.show()


if __name__ == '__main__':
    train_melody()
