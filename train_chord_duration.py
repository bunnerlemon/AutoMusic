import torch
import numpy as np

from until import *
from network import *

BATCH_SIZE = 128
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
NLAYERS = 10
GRAD_CLIP = 1
NUM_EPOCH = 20
USE_CUDA = torch.cuda.is_available()
DEVICE = ("cuda" if USE_CUDA else "cpu")


def train_chord_duration():
    train_chord_duration_input, train_chord_duration_output, num_chord_duration = prepare_train_chord_duration_data()
    val_chord_duration_input, val_chord_duration_output = prepare_val_chord_duration_data()
    model = ThreeLayerLSTM(num_chord_duration, EMBEDDING_SIZE, HIDDEN_SIZE, NLAYERS, dropout=0.5)
    if USE_CUDA:
        model = model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 0.2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    val_losses = []
    if os.path.exists("model/chord_duration_best.pth"):
        model.load_state_dict(torch.load("model/chord_duration_best.pth"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    for epoch in range(NUM_EPOCH):
        model.train()
        hidden = model.init_hidden(BATCH_SIZE)
        countx = 9

        for start in range(0, len(train_chord_duration_input) - BATCH_SIZE, BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = torch.LongTensor(train_chord_duration_input[start:end])
            batchY = torch.LongTensor(train_chord_duration_output[start:end])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 5), batchY.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            print("Epoch :", epoch, loss.item())
            if countx % 10 == 0:
                val_loss = evaluate(model, val_chord_duration_input, val_chord_duration_output, loss_fn, num_chord_duration)
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    val_losses.append(val_loss)
                    print("best_model_loss :", val_loss)
                    torch.save(model.state_dict(), "model/chord_duration_best.pth")
                else:
                    scheduler.step()
                    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
            countx += 1
                

def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden)


def evaluate(model, val_chord_duration_input, val_chord_duration_output, loss_fn, num_chord_duration):
    model.eval()
    total_loss = 0.
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for start in range(0, len(val_chord_duration_input) - BATCH_SIZE, BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = torch.LongTensor(val_chord_duration_input[start:end])
            batchY = torch.LongTensor(val_chord_duration_output[start:end])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 5), batchY.view(-1))
            total_count += float(len(val_chord_duration_input) * len(val_chord_duration_output))
            total_loss += float(loss.item()) * float(len(val_chord_duration_input) * len(val_chord_duration_output))
    loss = total_loss / total_count
    model.train()
    return loss


if __name__ == '__main__':
    train_chord_duration()