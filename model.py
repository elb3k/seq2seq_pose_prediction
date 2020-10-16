import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Reshape, Flatten


def seq2seq_model(SEQUENCES=75, JOINTS=17, DIMS=2, HIDDEN_DIM=[512, 256, 128], EMBEDDING_SIZE=128):
  
  LAYERS = len(HIDDEN_DIM)
  
  # Encoder
  encoder_inputs = Input(shape=(SEQUENCES, JOINTS, DIMS), dtype="float32")
  encoder_reshape = Reshape(target_shape=(SEQUENCES, JOINTS*DIMS)) (encoder_inputs)

  encoder_embedding = Dense(EMBEDDING_SIZE) (encoder_reshape)

  # Decoder
  decoder_inputs = Input(shape=(SEQUENCES, JOINTS, DIMS), dtype="float32")
  decoder_reshape = Reshape(target_shape=(
      SEQUENCES, JOINTS*DIMS))(decoder_inputs)



  decoder_embedding = Dense(EMBEDDING_SIZE)(decoder_reshape)


  # First LSTM
  encoder_LSTM = LSTM(HIDDEN_DIM[0], return_state=True, return_sequences=True)
  encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
  
  decoder_LSTM = LSTM(HIDDEN_DIM[0], return_state=True, return_sequences=True)
  decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

  # Last LSTMs
  for i in range(1, LAYERS):
    encoder_LSTM = LSTM(HIDDEN_DIM[i], return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_outputs)
    
    decoder_LSTM = LSTM(HIDDEN_DIM[i], return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_outputs, initial_state=[state_h, state_c])
  
  outputs = Flatten()(decoder_outputs)

  outputs = Dense(4096)(outputs)

  outputs = Dense(SEQUENCES*JOINTS*DIMS)(outputs)

  outputs = Reshape(target_shape=(SEQUENCES, JOINTS, DIMS))(outputs)

  model = Model([encoder_inputs, decoder_inputs], outputs)
  return model

