import torch
import numpy as np
import copy

from until import *
from network import *

BATCH_SIZE = 128
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
NUM_LAYERS = 10
GRAD_CLIP = 1
NUM_EPOCHS = 100
USE_CUDA = torch.cuda.is_available()


def train():

    train_sequence_input, train_sequence_output, num_pitch = prepare_train_sequence()
    val_sequence_input, val_sequenc_output = prepare_val_sequence()
    model = ThreeLayerLSTM(num_pitch, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout=0.3)
    if USE_CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.02
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    val_losses = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        hidden = model.init_hidden(BATCH_SIZE)
        countx = 0
        fail_countx = 0
        for start_idx in range(0, len(train_sequence_input) - BATCH_SIZE, BATCH_SIZE):
            end_idx = start_idx + BATCH_SIZE
            batchX = torch.LongTensor(train_sequence_input[start_idx:end_idx])
            batchY = torch.LongTensor(train_sequence_output[start_idx:end_idx])
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
            if countx % 50 == 0:
                val_loss = evaluate(model, val_sequence_input, val_sequenc_output, loss_fn)
                train_loss_list.append(loss.item())
                val_loss_list.append(val_loss)
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    print("best_model, val_loss", val_loss)
                    torch.save(model.state_dict(), "best_sequence.pth")
                    val_losses.append(val_loss)
                    fail_countx = 0
                else:
                    fail_countx += 1
                    if fail_countx == 3:
                        scheduler.step()
                        fail_countx = 2
            countx += 1


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def evaluate(model, val_sequence_input, val_sequence_output, loss_fn):
    model.eval()
    total_loss = 0.
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for start in range(0, len(val_sequence_input) - BATCH_SIZE, BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = torch.LongTensor(val_sequence_input[start:end])
            batchY = torch.LongTensor(val_sequence_output[start:end])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 10), batchY.view(-1))
            total_count += float(len(val_sequence_input) * len(val_sequence_output)) * 100.
            total_loss += float(loss.item()) * float(len(val_sequence_input) * 10. * float(len(val_sequence_output))) * 10.
    loss = total_loss / total_count
    model.train()
    return total_loss / total_count


if __name__ == '__main__':
    train()
