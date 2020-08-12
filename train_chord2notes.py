import torch
import numpy as np

from until import *
from network import *

BATCH_SIZE = 256
EMBEDDING_SIZE = 100
HIDDENG_SIZE = 100
NLAYERS = 20
NUM_DIRECTION = 2
GRAD_CLIP = 1
NUM_EPOCH = 50
USE_CUDA = torch.cuda.is_available()
DEVICE = ("cuda" if USE_CUDA else "cpu")


def train_chord2notes():

    if not os.path.exists("model/chord2notes"):
        os.mkdir("model/chord2notes")

    # 所有的和弦
    chords = read_("data/chord")
    chordnames = sorted(set(chords))
    chord2int = dict((chord, num) for num, chord in enumerate(chordnames))
    
    # 所有的音符
    if not os.path.exists("data/all_notes"):
        pure_notes()
    notes = read_("data/all_notes")
    notenames = sorted(set(notes))
    note2int = dict((note, num) for num, note in enumerate(notenames))
    num_note = len(notenames)

    # 对每个chord进行音符生成训练
    for i in range(len(chordnames)):
        if os.path.exists("model/chord2notes/"+str(i)+".pth"):
            continue
        # 取chord[i]对应的音符序列的训练和验证数据
        train_chord2notes_input, train_chord2notes_output = prepare_train_chord2notes_k(i)
        val_chord2notes_input, val_chord2notes_output = prepare_val_chord2notes_k(i)
        print("chord_idx:", i)
        # 对于音符数量过少的和弦，直接跳过
        if len(train_chord2notes_input) < BATCH_SIZE:
            continue
        print(len(train_chord2notes_input), len(train_chord2notes_output), len(val_chord2notes_input), len(val_chord2notes_output))
        model = GRU_BiDir(num_note, EMBEDDING_SIZE, HIDDENG_SIZE, NLAYERS, dropout=0.5)
        if USE_CUDA:
            model = model.cuda()

        loss_fn = torch.nn.CrossEntropyLoss()
        learning_rate = 0.2
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

        val_losses = []
        
        countx = 0
        MOD = 10
        NUM_EPOCH = 30
        if len(train_chord2notes_input) > 256000:
            NUM_EPOCH = 10
            MOD = 100
        elif len(train_chord2notes_input) > 25600:
            NUM_EPOCH = 20
            MOD = 50
        
        for epoch in range(NUM_EPOCH):
            model.train()
            hidden = model.init_hidden(BATCH_SIZE)
            countx = 0
            for start in range(0, len(train_chord2notes_input) - BATCH_SIZE, BATCH_SIZE):
                end = start + BATCH_SIZE
                batchX = torch.LongTensor(train_chord2notes_input[start:end])
                batchY = torch.LongTensor(train_chord2notes_output[start:end])
                if USE_CUDA:
                    batchX = batchX.cuda()
                    batchY = batchY.cuda()
                hidden = repackage_hidden(hidden)
                output, hidden = model(batchX, hidden)
                loss = loss_fn(output.view(-1, 20), batchY.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                print("Epoch :", epoch, loss.item())
                if not countx % MOD:
                    val_loss = evaluate(model, val_chord2notes_input, val_chord2notes_output, loss_fn, num_note)
                    if len(val_losses) == 0 or val_loss < min(val_losses):
                        print("best_chord2notes_model, val_loss", val_loss)
                        val_losses.append(val_loss)
                        if os.path.exists("model/chord2notes/" + str(i) + ".pth"):
                            model.load_state_dict(torch.load("model/chord2notes/" + str(i) + ".pth"))
                        torch.save(model.state_dict(), "model/chord2notes/" + str(i) + ".pth")
                    else:
                        scheduler.step()
                        optimizer = torch.optim.SGD(model.parameters(), learning_rate)
                countx += 1


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(h) for h in hidden)


def evaluate(model, val_chord2notes_input, val_chord2notes_output, loss_fn, num_note):
    model.eval()
    total_loss = 0.
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for start in range(0, len(val_chord2notes_input) - BATCH_SIZE, BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = torch.LongTensor(val_chord2notes_input[start:end])
            batchY = torch.LongTensor(val_chord2notes_output[start:end])
            if USE_CUDA:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(batchX, hidden)
            loss = loss_fn(output.view(-1, 20), batchY.view(-1))
            total_count += float(len(val_chord2notes_input) * len(val_chord2notes_output))
            total_loss += float(loss.item()) * float(len(val_chord2notes_input) * len(val_chord2notes_output))
    loss = total_loss / total_count
    model.train()
    return loss


if __name__ == '__main__':
    train_chord2notes()
