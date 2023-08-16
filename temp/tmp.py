algorithm = 'TCRL'
# dataset param
dataset = 'SV'  # KSC 176; SV 204; UP 103; IN 200
input_channel = 204
num_classes = 16
train_size = 20
patch_length = 11
root = './data'

noise_type = 'sym'
percent = 12
seed = 1

# model param
feature_dim = 64
# train param
batch_size = 64
num_workers = 1
lr = 0.0005
warmup_lr = 0.0005
epochs = 200
adjust_lr = 1
epoch_decay_start = 80
# For two-model algorithms
model1_type = 'MyNetV9_CL'
model2_type = 'MyNetV9_CL'
save_result = True

