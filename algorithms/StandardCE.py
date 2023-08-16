import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from losses import SCELoss, GCELoss, DMILoss
from tqdm import tqdm
from models import MyNet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
There may some error in this code, to obtain the accuracy, please modify the code 'HSI_dataset.py' line 182:
''torch_dataset_train = HSIDA(images=x_train, labels=torch.from_numpy(y_train).type(torch.LongTensor), mode='test')''
'''

class StandardCE:
    def __init__(
            self,
            config: dict = None,
            input_channel: int = 103,
            num_classes: int = 9,
    ):

        self.lr = config['lr']

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['epochs']
        self.beta1_plan = [mom1] * config['epochs']

        for i in range(config['epoch_decay_start'], config['epochs']):
            self.alpha_plan[i] = float(config['epochs'] - i) / (
                    config['epochs'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2

        self.device = device
        self.epochs = config['epochs']

        # scratch
        # self.model_scratch = network.SSRN_network(input_channel, num_classes).to(self.device)
        self.model_scratch = MyNet.SPAN(input_channel, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.adjust_lr = config['adjust_lr']

        # loss function
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes)
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes)
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes)

    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.model_scratch.eval()  # Change model to 'eval' mode
        gt_test = []
        pre = []
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            _, logits = self.model_scratch(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            gt_test.extend(np.array(labels))
            pre.extend(np.array(pred.cpu()))

        acc = 100 * float(correct) / float(total)
        return acc, pre, gt_test

    def train(self, train_loader, epoch):
        print('Training ...')
        train_acc_sum, n = 0.0, 0
        self.model_scratch.train()

        if self.adjust_lr == True:
            self.adjust_learning_rate(self.optimizer, epoch)

        pbar = tqdm(train_loader)
        for (images, labels) in pbar:
            x = Variable(images).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)

            _, logits = self.model_scratch(x)
            # loss_sup = F.cross_entropy(logits, labels)
            loss_sup = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            train_acc_sum += (logits.argmax(dim=1) == labels).sum().cpu().item()
            n += x.shape[0]
            pbar.set_description(
                'Epoch [%d/%d], loss_sup: %.4f, Train_acc: %.4f %%'
                % (epoch + 1, self.epochs, loss_sup.data.item(), train_acc_sum / n))

        return (train_acc_sum / n) * 100

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            # param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
