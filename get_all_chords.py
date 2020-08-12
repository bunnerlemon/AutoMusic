from until import *
from music21 import instrument, converter, note, chord, stream


def pure_chord():
    notes = get_notes()
    output_chord = []
    offset = 0
    for data in notes:
        # 若为和弦
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for note_chord in notes_in_chord:
                new_note = note.Note(int(note_chord))
                new_note.storedInsrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_chord.append(new_chord)
            offset += 0.5
    midi_stream = stream.Stream(output_chord)
    midi_stream.write("midi", fp="data/chord.mid")


if __name__ == '__main__':
    pure_chord()
