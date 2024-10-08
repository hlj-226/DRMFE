import argparse
import collections
import itertools
import torch
import random

from model import Drmfe
from get_indicator_matrix_A import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config
import time


def main(MR=[0.3]):
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['dataset'] = dataset
    print("Data set: " + config['dataset'])
    config['print_num'] = config['training']['epoch'] / 10     # print_num
    logger = get_logger()

    # Load data
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]

    for missingrate in MR:
        accumulated_metrics = collections.defaultdict(list)
        config['training']['missing_rate'] = missingrate
        print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
        for data_seed in range(1, args.test_time + 1):
            # get the mask
            np.random.seed(data_seed)
            mask = get_mask(2, x1_train_raw.shape[0], config['training']['missing_rate'])
            # mask the data
            x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
            x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

            x1_train = torch.from_numpy(x1_train).float().to(device)
            x2_train = torch.from_numpy(x2_train).float().to(device)
            mask = torch.from_numpy(mask).long().to(device)  # indicator matrix A

            # Set random seeds for model initialization
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

            # Build the model
            DRMFE = Drmfe(config)
            optimizer = torch.optim.Adam(
                itertools.chain(DRMFE.autoencoder1.parameters(), DRMFE.autoencoder2.parameters()),
                lr=config['training']['lr'])
            optimizer_pre = torch.optim.Adam(
                itertools.chain(DRMFE.autoencoder1.parameters(), DRMFE.autoencoder2.parameters()),
                lr=config['training']['lr'])
            DRMFE.to_device(device)

            # Training
            pre_loss = DRMFE.pretrain(config, X_list[0], X_list[1], optimizer_pre, device)
            acc, nmi, ari = DRMFE.train(config, logger, x1_train, x2_train, Y_list, mask, optimizer, device)
            accumulated_metrics['acc'].append(acc)
            accumulated_metrics['nmi'].append(nmi)
            accumulated_metrics['ari'].append(ari)
            print('------------------------Training over------------------------')
        cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])


if __name__ == '__main__':
    dataset = {3: "Scene-15"}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=str(3), help='dataset id')  # data index
    parser.add_argument('--test_time', type=int, default=str(1), help='number of test times')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = [0.3]
    if dataset == 'NoisyMNIST':
        MisingRate = [0.3]
    # main(MR=MisingRate)

    main(MR=MisingRate)
