# par_transfer.py - minimal complete parameter set for lowrank.py

# folder / IO
folder = "./test_rollout"

# identity
task = "RNNAuto"
classcomb = 0
whichset = "train"        # or "test"

# model / architecture
n_hidden = 30
n_latent = 10
k_steps = None

# data / sequence
L = 4
m = 2
alpha = 10
cue_size = 4
delay = 0

# training / experiment settings
n_epochs = 50
n_types = 2
frac_train = 0.8
loss = "CE"
out_bias = False
init = None
transfer = "relu"
datasplit = "0_0"
rank = None
train_noise = 0.0

# analysis
common_rotation = "hidden"   # "weights" or "hidden"

# misc
seed = 0

