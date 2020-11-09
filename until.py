import os
import numpy as np
import subprocess
import pickle
import glob
import random
from music21 import converter, instrument, note, chord, stream


def get_notes():

    # 从music_midi目录中的所有midi文件中提取notes(音符)和chord(和弦)
    # Note样例: A, B, A#, B#, G#, E, ...
    # chord样例: [B4, E5, G#5], [C5, E5], ...
    # 因为和弦就是多个note的集合，所以将其统称为“Note”
    
    # 确保包含所有Midi文件music_midi文件夹在所有python文件的同级目录下
    if not os.path.exists("piano_song"):
        raise Exception("包含所有MIDI的music_midi文件夹不在此目录下， 请添加")

    # 记录音符
    notes = []

    for midi_file in glob.glob("piano_song/*.MID"):
    # for midi_file in glob.glob("piano_song/001.MID"):
        stream = converter.parse(midi_file)
        parts = instrument.partitionByInstrument(stream)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = stream.flat.notes

        # 取音调，若是和弦，则转成音符
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # 保存所取的音调
    # 如果 data 目录不在，创建此目录
    if not os.path.exists("data"):
        os.mkdir("data")
    # 将数据写入 data 目录下的 notes 文件
    with open("data/notes", "wb") as filepath:
        pickle.dump(notes, filepath)
    return notes

def pure_chord():
    '''
    提取和弦, 分别存入训练集, 验证集
    对于所有的和弦序列，每五个和弦序列作为一批
    从每一批中随机抽取一个和弦序列加入验证集，剩余四个加入训练集
    其中chords存储的是所有的和弦
    分别存入 "data/chord/chords", "data/chord/train_chords", "data/chord/val_chords"
    '''
    if not os.path.exists("data/chord"):
        raise Exception("data/chord 路径不存在, 请先执行 data_pure.py")

    # 先从all_chords中读出所有的和弦序列
    all_chords = read_("data/chord/all_chords")

    chords = []
    train_chords = []
    val_chords = []

    for idx in range(0, len(all_chords), 5):
        end_idx = idx + 5
        if end_idx > len(all_chords):
            end_idx = len(all_chords)
        
        # 随机选取一条和弦序列加入验证集
        val_idx = random.randint(idx, end_idx - 1)
        for i in range(idx, end_idx):
            if i == val_idx:
                for data in all_chords[i]:
                    val_chords.append(data)
                    chords.append(data)
            else:
                for data in all_chords[i]:
                    train_chords.append(data)
                    chords.append(data)
    with open("data/chord/chords", "wb") as fp:
        pickle.dump(chords, fp)
    with open("data/chord/train_chord", "wb") as fp:
        pickle.dump(train_chords, fp)
    with open("data/chord/val_chord", "wb") as fp:
        pickle.dump(val_chords, fp)

# 将Music中所有midi的文件的note提取出来，并分为训练集和验证集
# 分别存入 "data/note/all_notes", "data/note/train_notes", "data/note/val_notes"
def pure_notes():
    '''
    将所有的音符序列提取出来，分为训练集和验证集
    每五条音符序列进行一次处理，每次从五条音符序列中随机抽取一条加入验证集，剩余的则加入训练集
    notes 存储了所有的音符
    三者分别存放于 "data/note/notes" "data/note/train_notes" "data/note/val_notes"
    '''
    if not os.path.exists("data/note"):
        raise Exception("请先运行data_pure.py")
    all_notes = read_("data/note/all_notes")
    notes = []
    train_notes = []
    val_notes = []
    countx = 0
    for idx in range(0, len(all_notes), 5):
        end_idx = idx + 5
        if end_idx > len(all_notes):
            end_idx = len(all_notes)

        # 随机选取一条音符序列加入验证集
        val_idx = random.randint(idx, end_idx-1)
        for i in range(idx, end_idx):
            if i == val_idx:
                for data in all_notes[i]:
                    val_notes.append(data)
                    notes.append(data)
            else:
                for data in all_notes[i]:
                    train_notes.append(data)
                    notes.append(data)
    if not os.path.exists("data/note"):
        os.mkdir("data/note")
    with open("data/note/notes", "wb") as fp:
        pickle.dump(notes, fp)
    with open("data/note/train_notes", "wb") as fp:
        pickle.dump(train_notes, fp)
    with open("data/note/val_notes", "wb") as fp:
        pickle.dump(val_notes, fp)


def pure_sequence():
    '''
    pure_sequence 用来从之前提取出的两条音轨中拆分出训练集和验证集  
    首先将所有音轨全部放入all_sequence中  
    接着从all_sequence中,　每五条sequence随机出一条sequence加入验证集，剩余的则加入训练集
    '''
    if not os.path.exists("data/melody/melodys"):
        raise Exception("请先运行data_pure.py")
    melodys = read_("data/melody/melody")
    songs = read_("data/song/song")
    all_sequences = []
    for melody in melodys:
        all_sequences.append(melody)
    for song in songs:
        all_sequences.append(song)
    sequences = []
    train_sequence = []
    val_sequence = []
    for idx in range(0, len(all_sequences), 5):
        end_idx = idx + 5
        if end_idx > len(all_sequences):
            end_idx = len(all_sequences)
        val_idx = random.randint(idx, end_idx - 1)
        for i in range(idx, end_idx):
            if i == val_idx:
                for data in all_sequences[i]:
                    sequences.append(data)
                    val_sequence.append(data)
            else:
                for data in all_sequences[i]:
                    sequences.append(data)
                    train_sequence.append(data)
    if not os.path.exists("data/sequence"):
        os.mkdir("data/sequence")
    with open("data/sequence/sequences", "wb") as fp:
        pickle.dump(sequences, fp)
    with open("data/sequence/train_sequence", "wb") as fp:
        pickle.dump(train_sequence, fp)
    with open("data/sequence/val_sequence", "wb") as fp:
        pickle.dump(val_sequence, fp)       


def read_(path):
    with open(path, "rb") as fp:
        out = pickle.load(fp)
    return out

# 准备训练和弦的训练数据
def prepare_train_chord():
    if not os.path.exists("data/chord/chords"):
        pure_chord()

    train_chords = read_("data/chord/train_chord")
    all_chords = read_("data/chord/chords")
    chord_names = sorted(set(all_chords))
    
    chord2int = dict((chord, num) for num, chord in enumerate(chord_names))
    train_input = []
    train_output = []
    sequence_len = 10
    for i in range(0, len(train_chords) - sequence_len - 1):
        sequence_in = train_chords[i:i+sequence_len]
        # sequence_out = train_chords[i+1:i+sequence_len+1]
        train_input.append([chord2int[data] for data in sequence_in])
        train_output.append(chord2int[train_chords[i+sequence_len]])
        # train_output.append([chord2int[data] for data in sequence_out])

    n_patterns = len(train_input)

    train_input = np.reshape(train_input, (n_patterns, sequence_len))
    train_input = train_input / float(len(chord_names))
    train_output = to_categorical(train_output, num_classes=len(chord_names))
    return train_input, train_output, len(chord_names)

# 准备训练和弦的验证数据
def prepare_val_chord():
    if not os.path.exists("data/chord"):
        pure_chord()
    val_chords = read_("data/chord/val_chord")
    all_chords = read_("data/chord/chords")
    chord_names = sorted(set(all_chords))
    
    chord2int = dict((chord, num) for num, chord in enumerate(chord_names))
    
    val_input = []
    val_output = []
    sequence_len = 10
    for i in range(0, len(val_chords) - sequence_len - 1):
        sequence_in = val_chords[i:i+sequence_len]
        # sequence_out = val_chords[i+1:i+sequence_len+1]
        val_input.append([chord2int[data] for data in sequence_in])
        val_output.append(chord2int[val_chords[i+sequence_len]])
        # val_output.append([chord2int[data] for data in sequence_out])
    
    n_patterns = len(val_input)
    val_input = np.reshape(val_input, (n_patterns, sequence_len))
    val_input = val_input / float(len(chord_names))
    val_output = to_categorical(val_output, num_classes=len(chord_names))
    return val_input, val_output


# 准备音符训练序列
def prepare_train_notes():
    '''
    prepare_train_notes 用于准备音符训练数据

    数据来源: data/note/train_notes
    
    sequence_len = 10
    '''
    if not os.path.exists("data/note/all_notes"):
        pure_notes()
    notes = read_("data/note/notes")
    note_names = sorted(set(notes))
    note2int = dict((note, num) for num, note in enumerate(note_names))
    train_notes = read_("data/note/train_notes")
    train_notes_input = []
    train_notes_output = []
    sequence_len = 10
    for i in range(0, len(train_notes) - sequence_len - 1):
        sequence_in = train_notes[i:i+sequence_len]
        train_notes_input.append([note2int[data] for data in sequence_in])
        train_notes_output.append(note2int[train_notes[i+sequence_len]])
    n_patterns = len(train_notes_input)
    train_notes_input = np.reshape(train_notes_input, (n_patterns, sequence_len))
    train_notes_input = train_notes_input / float(len(note_names))
    # 将输出整理成01矩阵
    train_notes_output = to_categorical(train_notes_output, num_classes=len(note_names))
    return train_notes_input, train_notes_output


def prepare_val_notes():
    '''
    prepare_val_notes 用于准备音符验证数据

    数据来源: data/note/val_notes
    
    sequence_len = 10
    '''
    if not os.path.exists("data/note"):
        pure_notes()
    notes = read_("data/note/notes")
    note_name = sorted(set(notes))
    note2int = dict((note, num) for num, note in enumerate(note_name))
    val_notes = read_("data/note/val_notes")
    val_notes_input = []
    val_notes_output = []
    sequence_len = 10
    for i in range(0, len(val_notes) - sequence_len - 1):
        sequence_in = val_notes[i:i+sequence_len]
        val_notes_input.append([note2int[data] for data in sequence_in])
        val_notes_output.append(note2int[val_notes[i+sequence_len]])
    n_patterns = len(val_notes_input)
    val_notes_input = np.reshape(val_notes_input, (n_patterns, sequence_len))
    val_notes_input = val_notes_input / float(len(note_name))
    val_notes_output = to_categorical(val_notes_output, num_classes=len(note_name))
    return val_notes_input, val_notes_output


def prepare_train_sequence():
    '''
    在此种数据处理方式中,　我们并没有将和弦与音符区分开来  
    prepare_train_sequence 准备训练用的序列  
    sequence_len = 10
    '''
    if not os.path.exists("data/sequence"):
        pure_sequence()

    sequences = read_("data/sequence/sequences")
    train_sequence = read_("data/sequence/train_sequence")

    pitch_names = sorted(set(sequences))
    pitch2int = dict((pitch_name, num) for num, pitch_name in enumerate(pitch_names))

    train_sequence_input = []
    train_sequence_output = []

    sequence_len = 10
    for i in range(0, len(train_sequence) - sequence_len - 1):
        sequence_in = train_sequence[i:i+sequence_len]
        train_sequence_input.append([pitch2int[data] for data in sequence_in])
        train_sequence_output.append(pitch2int[train_sequence[i+sequence_len]])
    n_patterns = len(train_sequence_input)
    train_sequence_input = np.reshape(train_sequence_input, (n_patterns, sequence_len))
    train_sequence_input = train_sequence_input / float(len(pitch_names))
    train_sequence_output = to_categorical(train_sequence_output, len(pitch_names))
    return train_sequence_input, train_sequence_output, len(pitch_names)


def prepare_val_sequence():
    '''
    prepare_val_sequence 准备验证序列  
    同样的, 此种方法中并没有将和弦与音符区分开来  
    sequence_len = 10
    '''
    if not os.path.exists("data/sequence"):
        pure_sequence()

    sequences = read_("data/sequence/sequences")
    val_sequence = read_("data/sequence/val_sequence")

    pitch_names = sorted(set(sequences))
    pitch2int = dict((pitch, num) for num, pitch in enumerate(pitch_names))

    val_sequence_input = []
    val_sequence_output = []

    sequence_len = 10
    for i in range(0, len(val_sequence) - sequence_len - 1):
        sequence_in = val_sequence[i:i+sequence_len]
        val_sequence_input.append([pitch2int[data] for data in sequence_in])
        val_sequence_output.append(pitch2int[val_sequence[i+sequence_len]])
    n_patterns = len(val_sequence_input)
    val_sequence_input = np.reshape(val_sequence_input, (n_patterns, sequence_len))
    val_sequence_input = val_sequence_input / float(len(pitch_names))
    val_sequence_output = to_categorical(val_sequence_output, len(pitch_names))
    return val_sequence_input, val_sequence_output
    

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
  

# if __name__ == '__main__':
    # if not os.path.exists("data/chord2notes"):
    #     pure_chord2notes()
    # train_chord2notes = read_("data/chord2notes/train_chord2notes")
    # val_chord2notes = read_("data/chord2notes/val_chord2notes")
    # notes = read_("data/all_notes")
    # note_names = sorted(set(notes))
    # print("note_names =", len(note_names))
    # print(len(train_chord2notes))
    # countx = 0
    # for notes in train_chord2notes:
    #     if len(notes) > 6400:
    #         countx += 1
    # print(countx)
    # countx = 0
    # for notes in val_chord2notes:
    #     if len(notes) < 640:
    #         countx += 1
    # print(countx)

