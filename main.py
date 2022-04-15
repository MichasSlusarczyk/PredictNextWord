import tensorflow
import pickle
import numpy as np
import os

from keras.optimizer_v2.adam import Adam
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical


def predict_next_words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    predictions = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == predictions:
            predicted_word = key
            break

    print(predicted_word)
    return predicted_word


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 0 dla gpu, -1 dla cpu

    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    if physical_devices:
        tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    file = open("Pride.txt", "r", encoding="utf8")

    # store file in list
    lines = []
    for i in file:
        lines.append(i)

    # Convert list to string
    data = ""
    for i in lines:
        data = ' '.join(lines)

    # replace unnecessary stuff with space
    data = \
        data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“', '').replace('”', '')
    # new line, carriage return, unicode character --> replace by space

    # remove unnecessary spaces
    data = data.split()
    data = ' '.join(data)
    # data[:500]

    # len(data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])

    # saving the tokenizer for predict function
    pickle.dump(tokenizer, open('token.pkl', 'wb'))

    sequence_data = tokenizer.texts_to_sequences([data])[0]
    # sequence_data[:15]

    # len(sequence_data)

    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    sequences = []

    for i in range(3, len(sequence_data)):
        words = sequence_data[i - 3:i + 1]
        sequences.append(words)

    print("The Length of sequences are: ", len(sequences))
    sequences = np.array(sequences)
    # sequences[:10]

    x = []
    y = []

    for i in sequences:
        x.append(i[0:3])
        y.append(i[3])

    x = np.array(x)
    y = np.array(y)

    print("Data: ", x[:10])
    print("Response: ", y[:10])

    y = to_categorical(y, num_classes=vocab_size)
    # y[:5]

    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=3))
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1000))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))

    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='plot.png', show_layer_names=True)

    checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
    model.fit(x, y, epochs=70, batch_size=64, callbacks=[checkpoint])

    # Load the model and tokenizer
    model = load_model('next_words.h5')
    tokenizer = pickle.load(open('token.pkl', 'rb'))

    while True:
        text = input("Enter your line: ")

        if text == "0":
            print("Execution completed.....")
            break

        else:
            try:
                text = text.split(" ")
                text = text[-3:]
                print(text)

                predict_next_words(model, tokenizer, text)

            except Exception as e:
                print("Error occurred: ", e)
                continue


if __name__ == "__main__":
    main()
