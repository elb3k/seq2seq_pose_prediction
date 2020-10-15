
from utils import *
from model import *

from tqdm import tqdm

batch_size = 64

model = seq2seq_model()

model.load_weights("weights/v1/weights_100.h5")

train_df, val_df, test_df = load_dataset()

test_x, test_y = preprocess(test_df)

count = len(test_y)

loss = []

for i in tqdm(range( int(np.ceil(count/batch_size)) ), desc="Testing"):
    loss.append( distance( model.predict(test_x[ i*batch_size: (i+1)*batch_size ]), test_y[ i*batch_size : (i+1)*batch_size ]) )

print("Loss: %f" % np.mean(loss) )
