from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(sorted(word_index.items(), key=lambda x: x[1]))  # sorte tuples of word_index; based on index(1) as a sorting key

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

padded_seq = pad_sequences(sequences, maxlen=max([len(x) for x in sequences]), padding='post')
print(padded_seq)
print(padded_seq.shape)

test_data = [
    'i really love my dog',
    'my dog loves my manatee '
]
test_seq = tokenizer.texts_to_sequences(test_data)

print(test_seq)
