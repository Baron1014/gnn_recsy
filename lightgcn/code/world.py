'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#args = parse_args()

ROOT_PATH = "/home/baron/HW/gnn_recsy/lightgcn/"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'movielens']
all_models  = ['mf', 'lgn']
config['batch_size'] = 4096
# config['bpr_batch_size'] = args.bpr_batch
# config['latent_dim_rec'] = args.recdim
# config['lightGCN_n_layers']= args.layer
# config['dropout'] = args.dropout
# config['keep_prob']  = args.keepprob
# config['A_n_fold'] = args.a_fold
# config['test_u_batch_size'] = args.testbatch
# config['multicore'] = args.multicore
# config['lr'] = args.lr
# config['decay'] = args.decay
# config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

config['bpr_batch_size'] = 2048
config['dropout'] = 0
config['keep_prob']  = 0.6
config['A_n_fold'] = 100
config['test_u_batch_size'] = 92
config['multicore'] = 0
config['pretrain'] = 0
config['A_split'] = False
config['bigdata'] = False
config['latent_dim_rec'] = 64
config['lightGCN_n_layers']=3
config['lr'] = 0.001
config['decay'] = 1e-4
seed = 2020
dataset = "movielens"
model_name = 'lgn'

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
# seed = args.seed

# dataset = args.dataset
# model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")





# TRAIN_epochs = args.epochs
# LOAD = args.load
# PATH = args.path
# topks = eval(args.topks)
# tensorboard = args.tensorboard
# comment = args.comment
TRAIN_epochs = 1000
LOAD = 0
PATH = "./checkpoints"
topks = eval("[20]")
tensorboard = 0
comment = 'lgn'
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
