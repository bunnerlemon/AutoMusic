import pickle
import os
import torch
import numpy as np

from network import *
from until import *

BATCH_SIZE = 128
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
GRAD_CLIP = 1
NUM_LAYERS = 10
NUM_EPOCHS = 100
USE_CUDA = torch.cuda.is_available()


def generate_sequence():
    notes = read_('data/sequence/sequences')
    
    pitch_name = sorted(set(notes))

    num_pitch = len(pitch_name)

    # network_input, normalize_input = prepare_sequences(notes, pitch_name, num_pitch)

    best_model = LSTM_BiDir(num_pitch, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout=0.5)

    if USE_CUDA:
        best_model = best_model.cuda()

    best_model.load_state_dict(torch.load("weight/best_sequence.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = best_model.init_hidden(1)

    input = torch.randint(num_pitch, (1, 1), dtype=torch.long).to(device)
    prediction = []

    int_to_pitch = dict((num, pitch) for num, pitch in enumerate(pitch_name))

    for i in range(100):
        output, hidden = best_model(input, hidden)
        output_weights = output.squeeze().exp().cpu()
        pitch_idx = torch.multinomial(output_weights, 1)[0]
        input.fill_(pitch_idx)
        pitch = int_to_pitch[int(pitch_idx.item())]
        prediction.append(pitch)
    
    create_music(prediction)
    # return prediction

def create_music(prediction):
    '''
    param prediction: 预测出来的音符序列

    offset = 0.5

    return : 将prediction中的音符按序组好放入 output/outputx.mid 中
    '''

    # 用神经网络预测的音乐数据生成Midi文件

    offset = 0
    output_notes = []
    pitch_duration = [500, 550, 600, 650, 690, 720]
    # 生成note 或 chord
    for data in prediction:
        # 若为chord
        if('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInsrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # 是 note
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += pitch_duration[random.randint(0, 5)] / 1000.

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output/output4.mid')

if __name__ == '__main__':
    generate_sequence()
        