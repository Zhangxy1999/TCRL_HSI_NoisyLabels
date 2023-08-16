algorithm = 'TCRL'
# dataset param
dataset = 'UP'  # KSC 176; SV 204; UP 103;
input_channel = 103   # channel of data
num_classes = 9     # number of class
train_size = 76  # Total number of samples in each class
patch_length = 11

noise_type = 'sym'  # 'sym' or 'asym'
percent = 24  # the number of noisy samples per class
seed = 1

# model param
feature_dim = 64
# train param
batch_size = 64   # batch size
num_workers = 1
lr = 0.0005  # learning rate
epochs = 200  # eppoch
adjust_lr = 1
epoch_decay_start = 80
# For two-model algorithms
model1_type = 'MyNetV9_CL'
model2_type = 'MyNetV9_CL'
save_result = True

