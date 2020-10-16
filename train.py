
from utils import *
from model import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import SGD


batch_size=64
epochs=100
learning_rate = 0.0001

model = seq2seq_model()

train_df, val_df, test_df = load_dataset()

train_x, train_y = preprocess(train_df)

val_x, val_y = preprocess(val_df)

optimizer = SGD(learning_rate)

# Model compile
model.compile(optimizer=optimizer, loss=loss)

tensorboard = TensorBoard(log_dir="log/v1")

checkpoint = ModelCheckpoint("weights/v1/weights_{epoch:03d}.h5")

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
  callbacks=[tensorboard, checkpoint],
  validation_data=(val_x, val_y), shuffle=True)
