import argparse
from datetime import datetime



def set_argparse_defs(parser):
    parser.set_defaults(accelerator='gpu')
    parser.set_defaults(devices=1)
    parser.set_defaults(num_sanity_val_steps=0)
    parser.set_defaults(deterministic=False)
    
    return parser



def add_argparse(parser):
    ### Data loader args
    parser.add_argument('--aug_config', dest='aug_config', default='dataset/augmentation_parameters.txt', \
                        help='text file listing augmentation parameters')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, \
                        help='# samples in batch')
    parser.add_argument('--data_config', dest='data_config', default='configs/data/data_config_t1.csv', \
                        help='text file listing input data')
    parser.add_argument('--n_workers', dest='n_workers', type=int, default=8, \
                        help='number of workers for data loaders')
    parser.add_argument('--output_dir', dest='output_dir', default=None, \
                        help='output folder for model predictions')

    ### Metrics args
    parser.add_argument('--metrics_train', dest='metrics_train', nargs='+', default=[], \
                        help='training metrics')
    parser.add_argument('--metrics_test', dest='metrics_test', nargs='+', default=[], \
                        help='testing metrics')
    parser.add_argument('--metrics_valid', dest='metrics_valid', nargs='+', default=[], \
                        help='validation metrics')

    ### Model args
    parser.add_argument('--activation_function', dest='activation_function', default='ELU', \
                        help='activation function for UNet')
    parser.add_argument('--n_levels', dest='n_levels', type=int, default=3, \
                        help='number of levels for UNet')
    parser.add_argument('--max_n_epochs', dest='max_n_epochs', type=int, default=1, \
                        help='number of epochs')
    parser.add_argument('--network', dest='network', default='UNet3D_3layers', \
                        help='training network')
    parser.add_argument('--pooling_function', dest='pooling_function', default='MaxPool', \
                        help='pooling function for UNet')

    ### Optimizer args
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, \
                        help='lower beta value for optimizer (Adam/AdamW)')
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.99, \
                        help='higher beta value for optimizer (Adam/AdamW)')
    parser.add_argument('--dampening', dest='dampening', type=float, default=0, \
                        help='dampening for optimizer (SGD)')
    parser.add_argument('--loss', dest='loss', default='dice_cce_loss', \
                        help='training loss function')
    parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.001, \
                        help='learning rate decay')
    parser.add_argument('--lr_scheduler', dest='lr_scheduler', default='LinearLR', \
                        help='lr_scheduler for optimizer (must be part of torch.optim')
    parser.add_argument('--lr_start', dest='lr_start', type=float, default=0.001, \
                        help='initial learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default='0', \
                        help='momentum for optimizer (SGD)')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam', \
                        help='optimizer for training (case-sensitive)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0, \
                        help='training weight decay')
    
    ### Seed
    parser.add_argument('--seed', dest='seed', type=int, default=0, \
                        help='random seed for xval')
    return parser
