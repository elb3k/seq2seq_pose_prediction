import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Reshape


def seq2seq_model(SEQUENCES=75, JOINTS=17, DIMS=2, HIDDEN_DIM=512, EMBEDDING_SIZE=512):
  # Encoder
  encoder_inputs = Input(shape=(SEQUENCES, JOINTS, DIMS), dtype="float32")
  encoder_reshape = Reshape(target_shape=(SEQUENCES, JOINTS*DIMS)) (encoder_inputs)

  encoder_embedding = Dense(EMBEDDING_SIZE) (encoder_reshape)
  encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
  encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

  # Decoder
  decoder_inputs = Input(shape=(SEQUENCES, JOINTS, DIMS), dtype="float32")
  decoder_reshape = Reshape(target_shape=(SEQUENCES, JOINTS*DIMS)) (decoder_inputs)

  decoder_embedding = Dense(EMBEDDING_SIZE) (decoder_reshape)
  decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
  decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
  outputs = TimeDistributed(Dense(JOINTS*DIMS))(decoder_outputs)

  outputs = Reshape(target_shape=(SEQUENCES, JOINTS, DIMS))(outputs)

  model = Model([encoder_inputs, decoder_inputs], outputs)
  return model

