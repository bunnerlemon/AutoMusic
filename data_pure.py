import os
import subprocess
import pickle
import glob

from music21 import converter, stream, instrument, note, chord


'''
midi原文件路径：music_data
对于任意一首Midi文件，若该文件有两个及以上track，则挑选出两个音符最多的track出来，否则仅取一条track
每一个track重新组成一个midi文件
目标文件路径：music/*.mid
按序号存储
'''


def data_pure():
    '''
    midi原文件路径：music_data  

    对于任意一首Midi文件，若该文件有两个及以上track，则挑选出两个音符最多的track出来，否则仅取一条track  
    对于挑选出来的track, 我们将其和弦序列与音符序列拆分出来  
    并且, 我们将每一个和弦后面的第一个音符也提取出来放入chord_notes中

    最终三个文件存放位置：
        "data/chord/all_chords"
        "data/note/all_notes"
        "data/chord_notes/chord_notes"
    '''
    if not os.path.exists("music_data"):
        raise Exception("当前目录下不存在文件夹music_data")
    # 初始的存储序号为0
    countx = 0
    all_chords = []
    all_notes = []
    chords_copy = []
    choose_id = []
    # 遍历music_data文件夹下所有的Midi文件
    for midi in glob.glob("music_data/*.mid"):
        midi_stream = converter.parse(midi)
        num_count = convert(midi_stream)
        print(num_count)
        choose = []
        id_1 = find_max_len(num_count, -1)
        if num_count[id_1]:
            choose.append(id_1)
        if len(num_count) > 1:
            id_2 = find_max_len(num_count, id_1)
            if num_count[id_2]:
                choose.append(id_2)
        choose_id.append(choose)
        # 将选择的轨道上的音符和和弦取出来
        for i in choose:
            notes = []
            chords = []
            parts = instrument.partitionByInstrument(midi_stream[i])
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi_stream[i].flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    chords.append('.'.join(str(data) for data in element.normalOrder))
                    chords_copy.append('.'.join(str(data) for data in element.normalOrder))
                    if('.'.join(str(data) for data in element.normalOrder) == '3.6.9'):
                        print("True")
            all_notes.append(notes)
            all_chords.append(chords)
    print(notes)
    print(chords)
    # return 0s
    # 提取每一个和弦后面的第一个音符放入chord_notes
    chord_names = sorted(set(chords_copy))
    chord2int = dict((chord, num) for num, chord in enumerate(chord_names))
    chord_notes = []
    for i in range(len(chord_names)):
        chord_notes.append([])
    for midi in glob.glob("music_data/*.mid"):
        midi_stream = converter.parse(midi)
        for i in choose_id[countx]:
            # print(i, countx)
            parts = instrument.partitionByInstrument(midi_stream[i])
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi_stream[i].flat.notes
            last_chord_idx = -1
            flag = False
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    if last_chord_idx == -1 or flag:
                        continue
                    else:
                        chord_notes[last_chord_idx].append(str(element.pitch))
                        flag = True
                elif isinstance(element, chord.Chord):
                    # print("chord=", '.'.join(str(data) for data in element.normalOrder))
                    last_chord_idx = chord2int['.'.join(str(data) for data in element.normalOrder)]
                    flag = False
        countx += 1

    if not os.path.exists("data/note"):
        os.mkdir("data/note")
    if not os.path.exists("data/chord"):
        os.mkdir("data/chord")
    if not os.path.exists("data/chord_notes"):
        os.mkdir("data/chord_notes")
    with open("data/note/all_notes", "wb") as fp:
        pickle.dump(all_notes, fp)
    with open("data/chord/all_chords", "wb") as fp:
        pickle.dump(all_chords, fp)
    with open("data/chord_notes/chord_notes", "wb") as fp:
        pickle.dump(chord_notes, fp)

# 将数据分为两个track, 一个旋律track, 一个正常音符track
def data2track():
    '''
    数据源：music_data/*.mid  
      
    在这种处理方法中,　我从每一个midi文件中提取出两条最长的音轨　　
    并且我将最长的音轨为melody, 第二条音轨为song
    '''
    if not os.path.exists("music_data"):
        raise Exception("当前目录下不存在Music data文件夹")
    melody = []
    notes = []
    for midi_file in glob.glob("music_data/*.mid"):
        midi_stream = converter.parse(midi_file)
        num_count = convert(midi_stream)
        choose = []
        # 找到第一条最长的音轨
        id_1 = find_max_len(num_count, -1)
        if num_count[id_1]:
            choose.append(id_1)
        if len(num_count) > 1:
            id_2 = find_max_len(num_count, id_1)
            if num_count[id_2]:
                choose.append(id_2)
        for i in range(len(choose)):
            parts = instrument.partitionByInstrument(midi_stream[choose[i]]) 
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi_stream[choose[i]].flat.notes
            if i == 0:
                track1 = []
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        track1.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        track1.append('.'.join(str(data) for data in element.normalOrder))
                melody.append(track1)
            else:
                track2 = []
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        track2.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        track2.append('.'.join(str(data) for data in element.normalOrder))
                notes.append(track2)
    if not os.path.exists("data/melody"):
        os.mkdir("data/melody")
    if not os.path.exists("data/song"):
        os.mkdir("data/song")
    with open("data/melody/melody", "wb") as fp:
        pickle.dump(melody, fp)
    with open("data/song/song", "wb") as fp:
        pickle.dump(notes, fp)


# 在stream里找到最大长度的非id的stream.part
def find_max_len(num_count, id):
    m_id = -1
    cur_len = -1
    for i in range(len(num_count)):
        if i == id:
            continue
        if m_id == -1:
            m_id = i
            cur_len = num_count[i]
        else:
            if num_count[i] > cur_len:
                m_id = i
                cur_len = num_count[i]
    return m_id


def convert(stream1):
    output = []
    for s in stream1:        
        choose = []
        id_1 = find_max_len
        # print(len(s))
        countx = 0
        parts = instrument.partitionByInstrument(s)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = s.flat.notes
        for element in notes_to_parse:
            # print(element)
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                countx += 1
        # print(countx)
        output.append(countx)
    return output


if __name__ == "__main__":
    data_pure()
    # data2track()
