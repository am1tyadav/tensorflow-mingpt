import os
import mingpt


filepath = "tmp/data.txt"

if not os.path.isfile(filepath):
    mingpt.data.download_example_data(filepath=filepath)

raw_data = mingpt.data.load_data(filepath=filepath)
vocab, encoder, decoder = mingpt.data.create_vocab(data=raw_data)
