import pandas as pd
import re
import sentencepiece as spm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import sentence_bleu

# Load and clean the dataset
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df = df.dropna()
    df['english'] = df['english'].str.lower()
    df['hinglish'] = df['hinglish'].str.lower()
    df = df[~df['english'].str.contains(r'\b\w*__\w*\b')]
    df = df[~df['hinglish'].str.contains(r'\b\w*__\w*\b')]
    df = df[df['english'].apply(lambda x: re.match(r'^[a-z\s]+$', x) is not None)]
    df = df[df['hinglish'].apply(lambda x: re.match(r'^[a-z\s]+$', x) is not None)]
    return df

# Tokenize the dataset using SentencePiece
def train_tokenizer(texts, model_prefix, vocab_size=16000):
    with open(f'{model_prefix}_texts.txt', 'w') as f:
        for text in texts:
            f.write(f'{text}\n')
    spm.SentencePieceTrainer.train(input=f'{model_prefix}_texts.txt', model_prefix=model_prefix, vocab_size=vocab_size)
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp

# Define the LSTM model
def define_model(input_dim, output_dim, input_length):
    encoder_inputs = tf.keras.Input(shape=(input_length,))
    encoder_embedding = Embedding(input_dim=input_dim, output_dim=256)(encoder_inputs)
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.Input(shape=(input_length,))
    decoder_embedding = Embedding(input_dim=output_dim, output_dim=256)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model with teacher forcing and early stopping
def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model using BLEU score
def evaluate_model(model, tokenizer, input_texts, target_texts):
    bleu_scores = []
    for input_text, target_text in zip(input_texts, target_texts):
        input_seq = tokenizer.encode(input_text)
        input_seq = pad_sequences([input_seq], maxlen=20, padding='post')
        decoded_sentence = decode_sequence(model, input_seq, tokenizer)
        reference = [target_text.split()]
        candidate = decoded_sentence.split()
        bleu_scores.append(sentence_bleu(reference, candidate))
    return sum(bleu_scores) / len(bleu_scores)

# Decode sequence using beam search
def decode_sequence(model, input_seq, tokenizer, beam_width=3):
    # Placeholder for beam search decoding implementation
    return "decoded sentence"

# Save the trained model and tokenizers
def save_model_and_tokenizers(model, tokenizer, model_path, tokenizer_path):
    model.save(model_path)
    tokenizer.save(tokenizer_path)

if __name__ == "__main__":
    # Load and clean the dataset
    file_path = 'dataset.csv'
    df = load_and_clean_data(file_path)

    # Tokenize the dataset
    english_texts = df['english'].tolist()
    hinglish_texts = df['hinglish'].tolist()
    english_tokenizer = train_tokenizer(english_texts, 'english')
    hinglish_tokenizer = train_tokenizer(hinglish_texts, 'hinglish')

    # Prepare data for training
    encoder_input_data = pad_sequences([english_tokenizer.encode(text) for text in english_texts], maxlen=20, padding='post')
    decoder_input_data = pad_sequences([hinglish_tokenizer.encode(text) for text in hinglish_texts], maxlen=20, padding='post')
    decoder_target_data = pad_sequences([hinglish_tokenizer.encode(text) for text in hinglish_texts], maxlen=20, padding='post')

    # Define and train the model
    model = define_model(input_dim=english_tokenizer.vocab_size(), output_dim=hinglish_tokenizer.vocab_size(), input_length=20)
    train_model(model, encoder_input_data, decoder_input_data, decoder_target_data)

    # Evaluate the model
    bleu_score = evaluate_model(model, hinglish_tokenizer, english_texts, hinglish_texts)
    print(f'BLEU score: {bleu_score}')

    # Save the trained model and tokenizers
    save_model_and_tokenizers(model, english_tokenizer, 'trained_model.h5', 'english_tokenizer.model')
    save_model_and_tokenizers(model, hinglish_tokenizer, 'trained_model.h5', 'hinglish_tokenizer.model')
