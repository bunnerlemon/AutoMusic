import torch
import random

from network import *
from until import *
from generate_chord import *

BATCH_SIZE = 128
NUM_LAYERS = 20
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"


def generate_notes():
    '''
    generate_notes 作为专门生成音符序列用

    其音符序列的生成依赖于已生成的和弦序列

    具体的生成流程：
        对于每一个和弦, 我们随机出来一个音符, 再按照(音符和弦比例)生成固定数量的音符
    '''
    # num 为用户指定的音符、和弦比例
    num = input("请输入音符:和弦比例：")
    num = int(num)
    
    # 从文件中读取所有的和弦 
    chords = read_("data/chord/chords")
    # 将所有的和弦去重并排序
    chord_names = sorted(set(chords))
    # 制作字典, {chord_name : idx}
    chord2int = dict((chord, num) for num, chord in enumerate(chord_names))

    # 从文件中读取所有的音符
    all_notes = read_("data/note/notes")
    # 将所有的音符去重并排序
    note_names = sorted(set(all_notes))
    # 制作字典, {idx : note}, 方便后面根据下标找音符
    int2note = dict((num, note) for num, note in enumerate(note_names))
    # 制作字典: {note : idx}
    note2int = dict((note, num) for num, note in enumerate(note_names))

    chord_notes = read_("data/chord_notes/chord_notes")
    # 预测得到的和弦序列
    prediction_chords = generate_chords()

    # 新建模型对象
    best_notes_model = ThreeLayerLSTM(len(note_names), EMBEDDING_SIZE, HIDDENG_SIZE, NUM_LAYERS, dropout=0.5)
    if USE_CUDA:
        best_notes_model = best_notes_model.cuda()

    # 将训练好的模型参数加载进来
    best_notes_model.load_state_dict(torch.load("model/notes.pth"))


    prediction_notes = []
    # 和弦与音符有关的写法
    # for chord in prediction_chords:
    #     idx = chord2int[chord]
    #     # print(idx)
    #     hidden = best_notes_model.init_hidden(1)
    #     if len(chord2notes[idx]) >= 64:
    #         rand_idx = random.randint(0, len(chord2notes[idx])-1)
    #         note_input = torch.LongTensor([[note2int[chord2notes[idx][rand_idx]]]]).to(DEVICE)
    #     prediction_notes_i = []
    #     if len(chord2notes[idx]) < 64:
    #         limit = 0
    #     elif len(chord2notes[idx]) < 1280:
    #         limit = 5
    #     else:
    #         limit = 8
    #     for i in range(0, limit, 1):
    #         note_output, hidden = best_notes_model(note_input, hidden)
    #         note_output_weights = note_output.squeeze().exp().cpu()
    #         note_idx = torch.multinomial(note_output_weights, 1)[0]
    #         note_input.fill_(note_idx)
    #         note = int2note[int(note_idx.item())]
    #         prediction_notes_i.append(note)
    #     prediction_notes.append(prediction_notes_i)

    # 单纯生成音符序列的写法
    # note_input = torch.randint(len(note_names), [1, 1], dtype=torch.long).to(DEVICE)
    # hidden = best_notes_model.init_hidden(1)
    # for chord in prediction_chords:
    #     chord_idx = chord2int[chord]
    #     if len(chord_notes[chord_idx]) < 2:
    #         prediction_notes.append("-1")
    #         continue
    #     rand_note = chord_notes[chord_idx][random.randint(0, len(chord_notes[chord_idx])-1)]
    #     rand_note_idx = note2int[rand_note]
    #     note_input = torch.randint(len(note_names), [1,1], dtype=torch.long).to(DEVICE)
    #     note_input.fill_(rand_note_idx)
    #     # note_input = torch.LongTensor(rand_note_idx, [1,1]).to(DEVICE)
    #     hidden = best_notes_model.init_hidden(1)
    #     for i in range(num):
    #         output, hidden = best_notes_model(note_input, hidden)
    #         output_weights = output.squeeze().exp().cpu()
    #         note_idx = torch.multinomial(output_weights, 1)[0]
    #         note_input.fill_(note_idx)
    #         note = int2note[int(note_idx.item())]
    #         prediction_notes.append(note)
    
    # 对于每一个和弦, 我们需要生成 num 个音符, 那么我们一共要生成 num * 30 个音符（这里我指定了和弦数量为30）
    # 随机生成一个音符
    note_input = torch.randint(len(note_names), [1,1], dtype=torch.long).to(DEVICE)
    hidden = best_notes_model.init_hidden(1)
    limit = 30 * num
    for i in range(limit):
        output, hidden = best_notes_model(note_input, hidden)
        output_weights = output.squeeze().exp().cpu()
        note_idx = torch.multinomial(output_weights, 1)[0]
        note_input.fill_(note_idx)
        note = int2note[int(note_idx.item())]
        prediction_notes.append(note)
    
    create_music(prediction_chords, prediction_notes, num)
    # return prediction_chords, prediction_notes


def create_music(prediction_chords, prediction_notes, num):
    '''
    param prediction_chords: 预测的和弦序列
    param prediction_notes: 预测的音符序列
    param num: 音符、和弦比例, 需要按比组装

    组装时的时长设置
    pitch_duration 是一个序列[400, 550, 700, 850]

    组装时随机生成下标取时长

    return: 生成的文件放入 output/musicx.mid 中
    '''
    print(prediction_notes)
    offset = 0
    pitch_duration = []
    chord_duration = []
    # 这里设置了按序递增的一个pitch_duration序列
    for i in range(400, 901, 150):
        pitch_duration.append(i)
    music = []
    countx = 0
    for predict_chord in prediction_chords:
        notes_in_chord = predict_chord.split('.')
        new_notes = []
        for cur_note in notes_in_chord:
            new_note = note.Note(int(cur_note))
            new_note.storedInstrument = instrument.Piano()
            new_notes.append(new_note)
        new_chord = chord.Chord(new_notes)
        # 设置和弦的偏移位置
        new_chord.offset = offset
        music.append(new_chord)
        flag = False
        # 对于每一个和弦，组装num个音符
        for i in range(countx, countx + num):
            if prediction_notes[i] == "-1":
                flag = True
                break
            new_note = note.Note(prediction_notes[i])
            new_note.offset = offset
            # 随机生成时长下标
            duration_idx = random.randint(0, 3)
            new_note.quarterLength = pitch_duration[duration_idx] / 1000.
            # 更新 offset
            offset += pitch_duration[duration_idx] / 1000.
            # 设置音符弹奏乐器为钢琴
            new_note.storedInstrument = instrument.Piano()
            music.append(new_note)
        if not flag:
            countx += num
        else:
            countx += 1

    # for i in range(100, 1001, 50):
    #     chord_duration.append(i)
    # music = []
    # countx = 0
    # for predict_chord in prediction_chords:
    #     # print(predict_chord, prediction_notes[countx])
    #     for predict_note in prediction_notes[countx]:
    #         new_note = note.Note(predict_note)
    #         new_note.offset = offset
    #         duration = pitch_duration[random.randint(0, len(pitch_duration) - 1)] / 1000.
    #         new_note.quarterLength = duration
    #         offset += duration
    #         new_note.storedInstrument = instrument.Piano()
    #         music.append(new_note)
    #     notes_in_chord = predict_chord.split('.')
    #     notes = []
    #     for cur_note in notes_in_chord:
    #         new_note = note.Note(int(cur_note))
    #         new_note.storedInstrument = instrument.Piano()
    #         notes.append(new_note)
    #     new_chord = chord.Chord(notes)
    #     new_chord.offset = offset
    #     duration = chord_duration[random.randint(0, len(chord_duration) - 1)] / 1000.
    #     new_chord.quarterLength = duration
    #     music.append(new_chord)
    #     offset += duration
    #     countx += 1
    # print(music)
    # stream1 = stream.Stream(chords)
    # stream2 = stream.Stream(notes)
    mid_stream = stream.Stream(music)
    # mid_stream.append(stream1)
    # mid_stream.append(stream2)
    # music.append(chords)
    # music.append(notes)
    # mid_stream = stream.Stream(music)
    # stream1.write("midi", fp="output/chord7.mid")
    # stream2.write("midi", fp="output/notes1.mid")
    mid_stream.write("midi", fp="output/music7.mid")
    print("generate success")
    # for element in music:
    #     print(element)


if __name__ == '__main__':
    generate_notes()
