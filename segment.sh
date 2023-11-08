#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00

set -x

# Define inputs
seed=0
data_config='dataset/data_config.csv'
aug_config='dataset/augmentation_parameters.txt'
batch_size=1
lr_start=0.001
lr_param=0.1
decay=0.0000
max_n_epochs=2000
optim='adam'
loss='dice_cce_loss'
metrics_train='MeanDice'
metrics_valid='MeanDice'
metrics_test='MeanDice'
schedule='poly'
output_dirs=('data/initial_test_elu_3layers' 'data/initial_test_relu_3layers' \
		 'data/initial_test_elu_4layers' 'data/initial_test_relu_4layers')

# Run train
network='UNet3D_'$SLURM_ARRAY_TASK_ID
output_dir=${output_dirs[$SLURM_ARRAY_TASK_ID]}

python3 train.py \
	--aug_config=$aug_config \
	--batch_size=$batch_size \
	--data_config=$data_config \
	--decay=$decay \
	--loss=$loss \
	--lr_param=$lr_param \
	--lr_start=$lr_start \
	--max_n_epochs=$max_n_epochs \
	--metrics_test=$metrics_test \
	--metrics_train=$metrics_train \
	--metrics_valid=$metrics_valid \
	--network=$network \
	--optim=$optim \
	--schedule=$schedule \
	--seed=$seed
