
from utils import *
from model import *

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

# Hyper Parameters
batch_size=64
epochs=100
learning_rate = 0.001

# Parameters
TIME_SEQUENCES = 75
HIDDEN_FEATURES = 256
DROPOUT = 0.5
NUM_STAGE = 12

# Path
log_dir = "log/v1"
weight_file = "weights/v1/weight_%03d.pth"

use_cuda = False
# Model
model = GCN(TIME_SEQUENCES, HIDDEN_FEATURES, DROPOUT, NUM_STAGE)

model = nn.DataParallel(model.cuda()) if use_cuda else model

# Load Dataset
train_df, val_df, test_df = load_dataset()

# Train Dataset
train_x, train_y = preprocess(train_df)

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Val Dataset
val_x, val_y = preprocess(val_df)

val_x = torch.Tensor(val_x)
val_y = torch.Tensor(val_y)

val_dataset = TensorDataset(val_x, val_y)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# SGD optimizer
optimizer = optim.SGD(learning_rate)

# Tensorboard writer
tensorboard = SummaryWriter(log_dir)

# Training
for epoch in range(epochs):

  train_loss = 0.0
  progressBar = tqdm(enumerate(train_loader), "Epochs: %03d, loss: %.5f" % (epoch+1, train_loss) )

  for i, (batch_x, batch_y) in progressBar:
    
    optimizer.zero_grad()

    loss = custom_loss(model(batch_x), batch_y)

    loss.backward()
    optimizer.step()

    train_loss += loss.data.item()

    progressBar.set_description("Epochs: %03d, loss: %.5f" % (epoch+1, loss.data.item()))

  train_loss /= (i+1)
  print("Train loss: ", train_loss)

  with torch.no_grad():
    val_loss = 0.0
    for i, (batch_x, batch_y) in enumerate(val_loader):
      
      loss = custom_loss(model(batch_x), batch_y)
      val_loss += loss.data.item()

    val_loss /= (i+1)
    print("Val loss: ", val_loss)


  # Tensorboard Writer
  tensorboard.add_scalar("loss/train", train_loss, epoch)
  tensorboard.add_scalar("loss/val", val_loss, epoch)

  # ModelCheckpoint
  torch.save(model.module.state_dict(), weight_file%(epoch+1))



