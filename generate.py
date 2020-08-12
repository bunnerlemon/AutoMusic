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
NUM_EPOCHS = 100
USE_CUDA = torch.cuda.is_available()


def generate():
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
    
    pitch_name = sorted(set(item for item in notes))

    num_pitch = len(pitch_name)

    network_input, normalize_input = prepare_sequences(notes, pitch_name, num_pitch)

    best_model = ThreeLayerLSTM(num_pitch, EMBEDDING_SIZE, HIDDEN_SIZE, 3, dropout=0.3)

    if USE_CUDA:
        best_model = best_model.cuda()

    best_model.load_state_dict(torch.load("lm_best.th"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = best_model.init_hidden(1)

    input = torch.randint(num_pitch, (1, 1), dtype=torch.long).to(device)
    prediction = []

    int_to_pitch = dict((num, pitch) for num, pitch in enumerate(pitch_name))

    for i in range(700):
        output, hidden = best_model(input, hidden)
        output_weights = output.squeeze().exp().cpu()
        pitch_idx = torch.multinomial(output_weights, 1)[0]
        input.fill_(pitch_idx)
        pitch = int_to_pitch[int(pitch_idx.item())]
        prediction.append(pitch)
    
    return prediction


def prepare_sequences(notes, pitch_name, num_pitch):
    sequence_length = 100

    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_name))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitch_to_int[i] for i in sequence_in])
        network_output.append(pitch_to_int[sequence_out])

    n_patterns = len(network_input)

    normalized_input = np.reshape(network_input, (n_patterns, sequence_length))

    normalized_input = normalized_input / float(num_pitch)

    return network_input, normalized_input

if __name__ == '__main__':
    generate()
        