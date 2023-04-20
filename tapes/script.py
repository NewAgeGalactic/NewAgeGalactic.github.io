from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

# Define the phonetic inventory
vowels = ['ae', 'eh', 'ie', 'ohe', 'we']
consonants = ['bc', 'cg', 'du', 'fur', 'ge', 'hn', 'ya', 'har', 'leh', 'meh', 'ne', 'pu', 'cue', 'ru', 'c', 'tra', 'vi', 'wi', 'xsh', 'whie', 'cha']

# Define the pronunciation rules
pronunciation = {'a': 'ay', 'e': 'eh', 'i': 'ie', 'o': 'oe', 'u': 'we', 'b': 'bs', 'c': 'se', 'd': 'du', 'f': 'fur',
                 'g': 'ge', 'h': 'hn', 'j': 'ya', 'k': 'vik', 'l': 'leh', 'm': 'meh', 'n': 'nuh', 'p': 'pu',
                 'q': 'cue', 'r': 'rue', 's': 'see',  'v': 'var', 'w': 'kik', 'x': 'xsh', 'y': 'wehi',
                 'z': ''}

# Create the input/output data
def create_data(num_examples):
    input_texts = []
    target_texts = []
    for i in range(num_examples):
        # Randomly generate a word of length 2-5
        word_len = np.random.randint(2, 6)
        word = ''
        for j in range(word_len):
            if j % 2 == 0:
                # Add a vowel
                word += np.random.choice(vowels)
            else:
                # Add a consonant
                word += np.random.choice(consonants)
        input_texts.append(word)
        # Reverse the word and apply the pronunciation rules
        target_text = ''.join([pronunciation[letter] for letter in word[::-1]])
        target_texts.append(target_text)
    return input_texts, target_texts

# Define the input/output sequence lengths
max_input_length = 5
max_target_length = 11

# Create the input/output tokenizers
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

# Create the data
num_examples = 10000
input_texts, target_texts = create_data(num_examples)

# Fit the tokenizers on the data
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

# Convert the input/output data to sequences
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Pad the input/output sequences
encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

# Shift the target sequences by one position
decoder_outputs = np.zeros_like(decoder_inputs)
decoder_outputs[:, 1:] = decoder_inputs[:, :-1]
decoder_outputs[:, 0] = target_tokenizer.word_index['\t']

# Define the model architecture
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1
latent_dim = 256

encoder_inputs_placeholder = tf.keras.layers.Input(shape=(max_input_length,))
x = tf.keras.layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs_placeholder)
x, state_h, state_c = tf.keras.layers.LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

decoder_inputs_placeholder = tf.keras.layers.Input(shape=(max_target_length,))
x = tf.keras.layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs_placeholder)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
x, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(x)

# Define the model and compile it
model = tf.keras.models.Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 100
model.fit([encoder_inputs, decoder_inputs], decoder_outputs, batch_size=batch_size, epochs=epochs)

# Define the encoder and decoder models for inference
encoder_model = tf.keras.models.Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = tf.keras.layers.Input(shape=(1,))
x = tf.keras.layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs_single)
decoder_outputs, state_h, state_c = decoder_lstm(x, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model([decoder_inputs_single] + decoder_states_inputs,
                                      [decoder_outputs] + decoder_states)

# Define the function to generate new words and their translations
def generate_word_and_translation():
    # Generate a new input sequence
    input_seq = np.zeros((1, max_input_length))
    # Randomly generate a word of length 2-5
    word_len = np.random.randint(2, 6)
    word = ''
    for j in range(word_len):
        if j % 2 == 0:
            # Add a vowel
            word += np.random.choice(vowels)
        else:
            # Add a consonant
            word += np.random.choice(consonants)
    # Convert the word to a sequence
    for t, char in enumerate(word):
        input_seq[0, t] = input_tokenizer.word_index[char]
    # Generate the translation
    decoded_sentence = ''
    # Initialize the decoder state with the encoder state
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['\t']
    for i in range(max_target_length - 1):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Get the next token and add it to the decoded sentence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit the loop if the end-of-sequence token is generated
        if sampled_char == '\n':
            break
        # Update the target sequence and decoder states
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return word, decoded_sentence[:-1]

# Generate and save 10 new words and their translations to a text file
with open('generated_words.txt', 'w') as f:
    for i in range(10):
        word, translation = generate_word_and_translation()
        f.write(f'{word} -> {translation}\n')