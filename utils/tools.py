import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from operator import truediv
from .config import Config
import random
import torch.backends.cudnn as cudnn
import json
import scipy.io as sio
import spectral as spy


def load_config(filename: str = None, _print: bool = True):
    '''
    load and print config
    '''
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict

    if _print == True:
        print_config(config)

    return config


def print_config(config):
    print('---------- params info: ----------')
    for k, v in config.items():
        print(k, ' : ', v)
    print('---------------------------------')


def get_log_name(day_str, config):
    # log_name =  config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
    #             str(config['percent']) + '_seed' + str(config['seed']) + '.json'
    log_name = config['dataset'] + day_str + '_' + config['algorithm'] + '_' + str(config['train_size']) + '_' + config[
        'noise_type'] + '_' + \
               str(config['percent']) + '_' + str(config['batch_size']) + '.json'
    if osp.exists('./log') is False:
        os.mkdir('./log')
    log_name = osp.join('./log', log_name)
    return log_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def plot_results(epochs, test_acc, plotfile):
    plt.style.use('ggplot')
    plt.plot(np.arange(1, epochs), test_acc, label='scratch - acc')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 20)))  # train epochs
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0, 101, 10))  # Acc range: [0, 100]
    plt.ylabel('Acc divergence')
    plt.savefig(plotfile)


def get_test_acc(acc):
    return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close()
    sio.savemat(name + '.mat', mdict={'Result': label})