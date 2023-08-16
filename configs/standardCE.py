algorithm = 'StandardCE'
# dataset param
dataset = 'UP'  # KSC 176; SV 204; UP 103;
input_channel = 103
num_classes = 9
train_size = 76
patch_length = 11

noise_type = 'sym'
percent = 24
seed = 1

loss_type = 'ce'
# model param
model1_type = 'MyNet'
# train param
batch_size = 64
lr = 0.0005
epochs = 200
num_workers = 0
adjust_lr = 1
epoch_decay_start = 80
# result param
save_result = True
