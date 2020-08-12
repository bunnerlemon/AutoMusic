from until import *
from mido import Message, MidiFile, MidiTrack


def output(notes):
    print(len(notes))
    for i in range(0, len(notes)):
        print(notes[i])


def view_tracks(filename):
    mid = MidiFile(filename)
    tracks = mid.tracks
    count = 0
    for track in tracks:
        print(count)
        for i in range(len(track)):
            # if "Melody" in str(track[i]):
            #     print(track[i])
                # return track
            print(track[i])
        count += 1
    print("num_tracks:", count)


def add_melody(track, filename):
    mid = MidiFile()
    mid.tracks.append(track)
    mid.save(filename)


if __name__ == '__main__':
    # output(get_notes())
    # add_melody(view_tracks("music_midi/feeling.mid"), "output/melody.mid")
    view_tracks("piano_song/001.MID")
