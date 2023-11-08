import argparse
from datetime import datetime



def set_argparse_defs(parser):
    parser.set_defaults(accelerator='gpu')
    parser.set_defaults(devices=1)
    parser.set_defaults(num_sanity_val_steps=0)
    parser.set_defaults(deterministic=False)

    return parser


def add_argparse_args(parser):
    parser.add_argument('--seed', dest='seed', type=int, default=0, \
                        help='random seed for xval')
    parser.add_argument('--data_config', dest='data_config', default='dataset/data_config.csv', \
                        help='text file listing input data')
    parser.add_argument('--aug_config', dest='aug_config', default='dataset/augmentation_parameters.txt', \
                        help='text file listing augmentation parameters')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, \
                        help='# samples in batch')
    parser.add_argument('--lr_start', dest='lr_start', type=float, default=0.0001, \
                        help='initial learning rate')
    parser.add_argument('--lr_param', dest='lr_param', type=float, default=0.1, \
                        help='final learning rate for sgd')
    parser.add_argument('--decay', dest='decay', type=float, default=0.0, \
                        help='training weight decay')
    parser.add_argument('--max_n_epochs', dest='max_n_epochs', type=int, default=1, \
                        help='number of epochs')
    parser.add_argument('--network', dest='network', default='UNet3D', \
                        help='training network')
    parser.add_argument('--optim', dest='optim', default='adam', \
                        help='optimizer for training')
    parser.add_argument('--loss', dest='loss', default='dice_cce_loss', \
                        help='training loss function')
    parser.add_argument('--metrics_train', dest='metrics_train', default='MeanDice', \
                        help='training metrics')
    parser.add_argument('--metrics_test', dest='metrics_test', default='MeanDice', \
                        help='testing metrics')
    parser.add_argument('--metrics_valid', dest='metrics_valid', default='MeanDice', \
                        help='validation metrics')
    parser.add_argument('--refresh_rate', dest='refresh_rate', type=int, default=1, \
                        help='refresh rate for pytorch lightning progress bar')
    parser.add_argument('--schedule', dest='schedule', default='flat', \
                        help='lr schedule policy')
    parser.add_argument('--output_dir', dest='output_dir', default=None, \
                        help='output folder for model predictions')
    return parser
