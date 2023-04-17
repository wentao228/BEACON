import argparse
import os

import numpy as np
import random
import warnings
import torch
import wandb
import itertools

warnings.filterwarnings("ignore")
from data_loader import DataLoader
from utils import get_logger
from model import Model
from configure import get_default_config


def set_seed(seed):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main(args, config, logger):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    seed = config['seed']  # seed = 14
    set_seed(seed)
    config['args_dataset'] = args.dataset
    config['args_ratio'] = args.ratio

    data = DataLoader(config)

    loss_model = Model(config)

    loss_model.to_device(device)

    print('start training....')

    lr = config['training']['lr']

    optimizer = torch.optim.Adam(
        itertools.chain(loss_model.compound_autoencoder.parameters(), loss_model.kg_autoencoder.parameters(),
                        loss_model.compound2kg.parameters(), loss_model.kg2compound.parameters(),
                        loss_model.protein_autoencoder.parameters(), loss_model.classifier.parameters(),
                        loss_model.protein2kg.parameters(), loss_model.kg2protein.parameters(),
                        ),
        lr=lr)

    cpi_result = loss_model.sl_train(data, config, logger, optimizer, device)
    print('success')

    return cpi_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--dataset', type=str, default='human', help='dataset')
    parser.add_argument('--ratio', type=str, default='random_1_1', help='ratio')
    parser.add_argument('--iteration', type=int, default='1', help='iteration')
    parser.add_argument('--is_debug', type=bool, default=False, help='is_debug')

    args = parser.parse_args()
    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
        wandb.init(mode="disabled")
    else:
        wandb.init(project='KC', config=args)
        wandb.run.log_code(".")
    print(args)

    results_cpi = []
    best_results_cpi = []
    config = get_default_config(args.dataset)
    logger = get_logger()
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = config['training']['seed'] + i
        config['seed'] = seed  # seed = 14
        config['iteration'] = i + 1
        cpi_r = main(args, config, logger)
        results_cpi.append(cpi_r)

    avg_cpi = np.mean(np.array(results_cpi), axis=0)
    std_cpi = np.std(results_cpi, axis=0)
    print('test results: ')
    print(avg_cpi)
    print(std_cpi)

    results_cpi.append(avg_cpi)
    results_cpi.append(std_cpi)

    results_parent_path = os.path.join(wandb.run.dir, 'results')
    if not os.path.exists(results_parent_path):
        os.mkdir(results_parent_path)
    np.savetxt('{}/cpi_{}_result.txt'.format(results_parent_path, args.dataset),
               np.array(results_cpi), delimiter=",", fmt='%f')

    print('result saved!!!')

    wandb.finish()
