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


def create_music(prediction):

    # 用神经网络预测的音乐数据生成Midi文件

    offset = 0
    output_notes = []

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

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output1.mid')

# 提取和弦, 分别存入训练集, 验证集, 训练集大小为前240首曲子
# 分别存入 "data/chord/chords", "data/chord/train_chords", "data/chord/val_chords"
def pure_chord():
    if not os.path.exists("data/chord"):
        raise Exception("data/chord 路径不存在, 请先执行 data_pure.py")
    all_chords = read_("data/chord/all_chords")
    chords = []
    train_chords = []
    val_chords = []
    for idx in range(0, len(all_chords), 5):
        end_idx = idx + 5
        if end_idx > len(all_chords):
            end_idx = len(all_chords)
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


# 提取和弦的时长, 同上分为训练集, 验证集, 训练集大小为前200首曲子
def pure_chord_duration():
    if not os.path.exists("music"):
        raise Exception("路径不存在")

    all_chord_duration = []
    train_chord_duration = []
    val_chord_duration = []
    countx = 0
    for midi_file in glob.glob("music/*.mid"):
        stream = converter.parse(midi_file)
        parts = instrument.partitionByInstrument(stream)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = stream.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, chord.Chord):
                if countx < 200:
                    train_chord_duration.append(element.quarterLength)
                else:
                    val_chord_duration.append(element.quarterLength)
                all_chord_duration.append(element.quarterLength)
        countx += 1
        
    if not os.path.exists("data/duration"):
        os.mkdir("data/duration")
    with open("data/duration/all_chord_duration", "wb") as fp:
        pickle.dump(all_chord_duration, fp)
    with open("data/duration/train_chord_duration", "wb") as fp:
        pickle.dump(train_chord_duration, fp)
    with open("data/duration/val_chord_duration", "wb") as fp:
        pickle.dump(val_chord_duration, fp)

    
# 处理出和弦以及对应的音符序列
def pure_chord2notes():
    if not os.path.exists("music"):
        raise Exception("不存在该来路径")
    if not os.path.exists("data/chord"):
        pure_chord()
    
    all_chords = read_("data/chord/chords")
    chord_names = sorted(set(all_chords))
    chord2int = dict((chord, num) for num, chord in enumerate(chord_names))

    train_chord2notes = []
    val_chord2notes = []
    chord2notes = []
    for i in range(len(chord_names)):
        train_chord2notes.append([])
        val_chord2notes.append([])
        chord2notes.append([])
    for midi_file in glob.glob("piano_song/*.MID"):
        stream = converter.parse(midi_file)
        parts = instrument.partitionByInstrument(stream)
        
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = stream.flat.notes

        notes = []
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            else:
                chord = '.'.join(str(data) for data in element.normalOrder)
                for data in notes:
                    chord2notes[chord2int[chord]].append(data)
    countx = 0
    for notes in chord2notes:
        if len(notes) > 100:
            limit = (len(notes) * 4) // 5
            for i in range(0, len(notes)):
                if i < limit:
                    train_chord2notes[countx].append(notes[i])
                else:
                    val_chord2notes[countx].append(notes[i])
        countx += 1
    if not os.path.exists("data/chord2notes"):
        os.mkdir("data/chord2notes")
    with open("data/chord2notes/train_chord2notes", "wb") as fp:
        pickle.dump(train_chord2notes, fp)
    with open("data/chord2notes/val_chord2notes", "wb") as fp:
        pickle.dump(val_chord2notes, fp)

# 将Music中所有midi的文件的note提取出来，并分为训练集和验证集
# 分别存入 "data/note/all_notes", "data/note/train_notes", "data/note/val_notes"
def pure_notes():
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


# 将"data/melody/melody"中的数据分成训练集和验证集
def pure_melody():
    if not os.path.exists("data/melody/melody"):
        raise Exception("请先运行data_pure.py")
    melody = read_("data/melody/melody")
    train_melody = []
    val_melody = []
    all_melody = []
    for idx in range(0, len(melody), 5):
        end_idx = idx + 5
        if end_idx > len(melody):
            end_idx = len(melody)
        val_idx = random.randint(idx, end_idx - 1)
        for i in range(idx, end_idx):
            if i == val_idx:
                for data in melody[i]:
                    val_melody.append(data)
                    all_melody.append(data)
            else:
                for data in melody[i]:
                    train_melody.append(data)
                    all_melody.append(data)
    with open("data/melody/melodys", "wb") as fp:
        pickle.dump(all_melody, fp)
    with open("data/melody/train_melody", "wb") as fp:
        pickle.dump(train_melody, fp)
    with open("data/melody/val_melody", "wb") as fp:
        pickle.dump(val_melody, fp)

    
def pure_song():
    if not os.path.exists("data/song"):
        raise Exception("请先运行data_pure.py")
    all_song = read_("data/song/song")
    train_song = []
    val_song = []
    songs = []
    for idx in range(0, len(all_song), 5):
        end_idx = idx + 5
        if end_idx > len(all_song):
            end_idx = len(all_song)
        val_idx = random.randint(idx, end_idx - 1)
        for i in range(idx, end_idx):
            if i == val_idx:
                for data in all_song[idx]:
                    val_song.append(data)
                    songs.append(data)
            else:
                for data in all_song[idx]:
                    train_song.append(data)
                    songs.append(data)
    with open("data/song/songs", "wb") as fp:
        pickle.dump(songs, fp)
    with open("data/song/train_song", "wb") as fp:
        pickle.dump(train_song, fp)
    with open("data/song/val_song", "wb") as fp:
        pickle.dump(val_song)


def pure_sequence():
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
def prepare_train_data():
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
def prepare_val_data():
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


# 获取时长训练数据与总体的时长总数
def prepare_train_chord_duration_data():
    if not os.path.exists("data/duration"):
        pure_chord_duration()

    train_chord_duration = read_("data/duration/train_chord_duration")
    all_chord_duration = read_("data/duration/all_chord_duration")
    chord_duration_names = sorted(set(all_chord_duration))
    chord_duration2int = dict((chord_duration, num) for num, chord_duration in enumerate(chord_duration_names))

    train_chord_duration_input = []
    train_chord_duration_output = []
    sequence_len = 5
    for i in range(0, len(train_chord_duration) - sequence_len - 1):
        sequence_in = train_chord_duration[i:i+sequence_len]
        train_chord_duration_input.append([chord_duration2int[data] for data in sequence_in])
        train_chord_duration_output.append(chord_duration2int[train_chord_duration[i+sequence_len]])
    n_patterns = len(train_chord_duration_input)
    train_chord_duration_input = np.reshape(train_chord_duration_input, (n_patterns, sequence_len))
    train_chord_duration_input = train_chord_duration_input / float(len(chord_duration_names))
    train_chord_duration_output = to_categorical(train_chord_duration_output, num_classes=len(chord_duration_names))
    
    return train_chord_duration_input, train_chord_duration_output, len(chord_duration_names)

# 获取验证集时长数据
def prepare_val_chord_duration_data():
    if not os.path.exists("data/duration"):
        pure_chord_duration()
    
    val_chord_duration = read_("data/duration/val_chord_duration")
    all_chord_duration = read_("data/duration/all_chord_duration")

    chord_duration_names = sorted(set(all_chord_duration))
    chord_duration2int = dict((chord_duration, num) for num, chord_duration in enumerate(chord_duration_names))

    val_chord_duration_input = []
    val_chord_duration_output = []
    sequence_len = 5
    for i in range(0, len(val_chord_duration) - sequence_len - 1):
        sequence_in = val_chord_duration[i:i+sequence_len]
        val_chord_duration_input.append([chord_duration2int[data] for data in sequence_in])
        val_chord_duration_output.append(chord_duration2int[val_chord_duration[i+sequence_len]])
    n_patterns = len(val_chord_duration_input)
    val_chord_duration_input = np.reshape(val_chord_duration_input, (n_patterns, sequence_len))
    val_chord_duration_input = val_chord_duration_input / float(len(chord_duration_names))
    val_chord_duration_output = to_categorical(val_chord_duration_output, num_classes=len(chord_duration_names))

    return val_chord_duration_input, val_chord_duration_output

# 准备第k个和弦-音符序列数据
def prepare_train_chord2notes_k(k):
    # 若文件已存在，直接读取即可
    # if os.path.exists("data/chord2notes/train_chord2notes_input") and os.path.exists("data/chord2notes/train_chord2notes_output"):
    #     return read_("data/chord2notes/train_chord2notes_input"), read_("data/chord2notes/train_chord2notes_output")
    if os.path.exists("data/chord2notes/train_chord2notes_" + str(k)):
        return read_("data/chord2notes/train_chord2notes_" + str(k) + "/train_chord2notes_input"), read_("data/chord2notes/train_chord2notes_" + str(k) + "/train_chord2notes_output")

    # 若和弦到音符序列未提取出来过，需要提取
    if not os.path.exists("data/chord2notes"):
        pure_chord2notes()
    # 提取所有音符
    if not os.path.exists("data/all_notes"):
        pure_notes()
    train_chord2notes = read_("data/chord2notes/train_chord2notes")
    all_chords = read_("data/chord")
    chord_names = sorted(set(all_chords))
    chord2int = dict((chord, num) for num, chord in enumerate(chord_names))
    all_notes = read_("data/all_notes")
    note_names = sorted(set(all_notes))
    note2int = dict((note, num) for num, note in enumerate(note_names))
    
    train_chord2notes_input = []
    train_chord2notes_output = []
    notes = train_chord2notes[k]
    if len(notes) >= 512:
        multi_num = 2560 // len(notes)
        if multi_num > 1:
            length = len(notes)
            for i in range(multi_num - 1):
                for j in range(length):
                    notes.append(notes[i])
        sequence_len = 20
        for i in range(0, len(notes) - sequence_len - 1):
            sequence_in = [note2int[data] for data in notes[i:i+sequence_len]]
            train_chord2notes_input.append(sequence_in)
            train_chord2notes_output.append(note2int[notes[i+sequence_len]])
        train_chord2notes_input = np.reshape(train_chord2notes_input, (len(train_chord2notes_input), sequence_len))
        train_chord2notes_input = train_chord2notes_input / float(len(note_names))
        train_chord2notes_output = to_categorical(train_chord2notes_output, num_classes=len(note_names))

    if not os.path.exists("data/chord2notes/train_chord2notes_"+str(k)):
        os.mkdir("data/chord2notes/train_chord2notes_"+str(k))
    with open("data/chord2notes/train_chord2notes_" + str(k) + "/train_chord2notes_input", "wb") as fp:
        pickle.dump(train_chord2notes_input, fp)
    with open("data/chord2notes/train_chord2notes_" + str(k) + "/train_chord2notes_output", "wb") as fp:
        pickle.dump(train_chord2notes_output, fp)
    return train_chord2notes_input, train_chord2notes_output


def prepare_val_chord2notes_k(k):
    if os.path.exists("data/chord2notes/val_chord2notes_"+str(k)):
        return read_("data/chord2notes/val_chord2notes_"+str(k)+"/val_chord2notes_input"), read_("data/chord2notes/val_chord2notes_"+str(k)+"/val_chord2notes_output")
    if not os.path.exists("data/chord2notes"):
        pure_chord2notes()
    if not os.path.exists("data/all_notes"):
        pure_notes()

    val_chord2notes = read_("data/chord2notes/val_chord2notes")
    
    all_notes = read_("data/all_notes")
    note_names = sorted(set(all_notes))
    note2int = dict((note, num) for num, note in enumerate(note_names))

    val_chord2notes_input = []
    val_chord2notes_output = []
    notes = val_chord2notes[k]
    
    if len(notes) >= 128:
        multi_num = 1280 // len(notes)
        if multi_num > 1:
            length = len(notes)
            for i in range(multi_num - 1):
                for j in range(length):
                    notes.append(notes[j])
        sequence_len = 20
        for i in range(0, len(notes) - sequence_len - 1):
            sequence_in = [note2int[data] for data in notes[i:i+sequence_len]]
            val_chord2notes_input.append(sequence_in)
            val_chord2notes_output.append(note2int[notes[i+sequence_len]])
        val_chord2notes_input = np.reshape(val_chord2notes_input, (len(val_chord2notes_input), sequence_len))
        val_chord2notes_input = val_chord2notes_input / float(len(note_names))
        val_chord2notes_output = to_categorical(val_chord2notes_output, num_classes=len(note_names))
        

    if not os.path.exists("data/chord2notes/val_chord2notes_"+str(k)):
        os.mkdir("data/chord2notes/val_chord2notes_"+str(k))
    with open("data/chord2notes/val_chord2notes_"+str(k)+"/val_chord2notes_input", "wb") as fp:
        pickle.dump(val_chord2notes_input, fp)
    with open("data/chord2notes/val_chord2notes_"+str(k)+"/val_chord2notes_output", "wb") as fp:
        pickle.dump(val_chord2notes_output, fp)
    return val_chord2notes_input, val_chord2notes_output

# 准备音符训练序列
def prepare_train_notes():
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
    train_notes_output = to_categorical(train_notes_output, num_classes=len(note_names))
    return train_notes_input, train_notes_output


def prepare_val_notes():
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


def prepare_train_melody():
    if not os.path.exists("data/melody/melodys"):
        pure_melody()
    
    melodys = read_("data/melody/melodys")
    melody_names = sorted(set(melodys))
    melody2int = dict((melody, num) for num, melody in enumerate(melody_names))

    train_melody = read_("data/melody/train_melody")

    melody_train_input =[]
    melody_train_output = []
    
    sequence_len = 10

    for i in range(0, len(train_melody) - sequence_len - 1):
        sequence_in = train_melody[i:i+sequence_len]
        melody_train_input.append([melody2int[data] for data in sequence_in])
        melody_train_output.append(melody2int[train_melody[i+sequence_len]])
    n_patterns = len(melody_train_input)
    melody_train_input = np.reshape(melody_train_input, (n_patterns, sequence_len))
    melody_train_input = melody_train_input / float(len(melody_names))
    melody_train_output = to_categorical(melody_train_output, len(melody_names))
    return melody_train_input, melody_train_output


def prepare_val_melody():
    if not os.path.exists("data/melody/melodys"):
        pure_melody()
    
    melodys = read_("data/melody/melodys")
    melody_names = sorted(set(melodys))
    melody2int = dict((melody, num) for num, melody in enumerate(melody_names))

    val_melody = read_("data/melody/val_melody")

    melody_val_intput = []
    melody_val_output = []

    sequence_len = 10
    
    for i in range(0, len(val_melody) - sequence_len - 1):
        sequence_in = val_melody[i:i+sequence_len]
        melody_val_intput.append([melody2int[data] for data in sequence_in])
        melody_val_output.append(melody2int[val_melody[i+sequence_len]])
    n_patterns = len(melody_val_intput)
    melody_val_intput = np.reshape(melody_val_intput, (n_patterns, sequence_len))
    melody_val_intput = melody_val_intput / float(len(melody_names))
    melody_val_output = to_categorical(melody_val_output, len(melody_names))
    return melody_val_intput, melody_val_output


def prepare_train_song():
    if not os.path.exists("data/song/songs"):
        pure_song()
    
    songs = read_("data/song/songs")
    song_names = sorted(set(songs))
    song2int = dict((song, num) for num, song in enumerate(song_names))

    train_song = read_("data/song/train_song")

    song_train_input = []
    song_train_output = []

    sequence_len = 10

    for i in range(0, len(train_song) - sequence_len - 1):
        sequence_in = train_song[i:i+sequence_len]
        song_train_input.append([song2int[data] for data in sequence_in])
        song_train_output.append(song2int[train_song[i+sequence_len]])
    n_patterns = len(song_train_input)
    song_train_input = np.reshape(song_train_input, (n_patterns, sequence_len))
    song_train_input = song_train_input / float(len(song_names))
    song_train_output = to_categorical(song_train_output, len(song_names))
    return song_train_input, song_train_output


def prepare_val_song():
    if not os.path.exists("data/song/songs"):
        pure_song()
    
    songs = read_("data/song/songs")
    song_names = sorted(set(songs))
    song2int = dict((song, num) for num, song in enumerate(song_names))

    val_song = read_("data/song/val_song")

    song_val_input = []
    song_val_output = []

    sequence_len = 10

    for i in range(0, len(val_song) - sequence_len - 1):
        sequence_in = val_song[i:i+sequence_len]
        song_val_input.append([song2int[data] for data in sequence_in])
        song_val_output.append(song2int[val_song[i+sequence_len]])
    
    n_patterns = len(song_val_input)
    song_val_input = np.reshape(song_val_input, (n_patterns, sequence_len))
    song_val_input = song_val_input / float(len(song_names))
    song_val_output = to_categorical(song_val_output, len(song_names))
    return song_val_input, song_val_output


# 最后一种方式
def prepare_train_sequence():
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
  

if __name__ == '__main__':
    if not os.path.exists("data/chord2notes"):
        pure_chord2notes()
    train_chord2notes = read_("data/chord2notes/train_chord2notes")
    val_chord2notes = read_("data/chord2notes/val_chord2notes")
    notes = read_("data/all_notes")
    note_names = sorted(set(notes))
    print("note_names =", len(note_names))
    print(len(train_chord2notes))
    countx = 0
    for notes in train_chord2notes:
        if len(notes) > 6400:
            countx += 1
    print(countx)
    countx = 0
    for notes in val_chord2notes:
        if len(notes) < 640:
            countx += 1
    print(countx)

