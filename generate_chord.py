import pickle
import os
import torch
import numpy as np

from network import *
from until import *

BATCH_SIZE = 128
EMBEDDING_SIZE = 20
HIDDENG_SIZE = 100
NLAYERS = 10
GRAD_CLIP = 1
NUM_EPOCH = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = ("cuda" if USE_CUDA else "cpu")


def generate_chords():
    all_chord = read_("data/chord/chords")
    chord_names = sorted(set(all_chord))
    num_chord = len(chord_names) 

    best_chord_model = LSTM_BiDir(num_chord, EMBEDDING_SIZE, HIDDENG_SIZE, NLAYERS, dropout=0.5)

    if USE_CUDA:
        best_chord_model = best_chord_model.cuda()

    best_chord_model.load_state_dict(torch.load("model/chord_best.pth"))

    chord_hidden = best_chord_model.init_hidden(1)

    chord_input = torch.randint(num_chord, (1, 1), dtype=torch.long).to(DEVICE)
    prediction_chords = []
    int2chord = dict((num, chord) for num, chord in enumerate(chord_names))

    for i in range(30):
        chord_output, chord_hidden = best_chord_model(chord_input, chord_hidden)
        chord_output_weights = chord_output.squeeze().exp().cpu()
        chord_idx = torch.multinomial(chord_output_weights, 1)[0]
        chord_input.fill_(chord_idx)
        chord = int2chord[int(chord_idx.item())]
        prediction_chords.append(chord)
        # print(chord_input)
    
    # create_chords(prediction_chords)
    return prediction_chords


def create_chords(prediction_chords):
    # print(prediction_chord_duration)
    offset = 0
    output_chords = []

    for data in prediction_chords:
        notes_in_chord = data.split('.')
        notes = []
        for cur_note in notes_in_chord:
            new_note = note.Note(int(cur_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes=notes)
        new_chord.offset = offset
        new_chord.quarterLength = 0.72
        output_chords.append(new_chord)
        offset += 0.72

    mid_stream = stream.Stream(output_chords)
    mid_stream.write("midi", fp="output/chord6.mid")


if __name__ == '__main__':
    generate_chords()
