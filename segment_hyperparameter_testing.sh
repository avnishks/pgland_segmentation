#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ppseg

run_type=slurm
#run_type=debug

n_start=3
n_jobs=$(find hyperparameter_configs/hyperparameter_options_*.txt | wc -l)
n_workers=4

set -x

function call-train(){
    if [ $run_type == "slurm" ] ; then
	JOB_ID=$SLURM_ARRAY_TASK_ID
    else
	JOB_ID=$1
    fi
    
    # Parse hyperparameters
    hyperparam_file="hyperparameter_configs/hyperparameter_options_${JOB_ID}.txt"
    lr_start=$(sed '1q;d' $hyperparam_file)
    lr_param=$(sed '2q;d' $hyperparam_file)
    decay=$(sed '3q;d' $hyperparam_file)
    schedule=$(sed '4q;d' $hyperparam_file)
    
    # Define inputs
    seed=0
    data_config='dataset/data_config_t1.csv'
    aug_config='dataset/augmentation_parameters.txt'
    batch_size=1
    max_n_epochs=2000
    network='UNet3D'
    optim='adam'
    loss='dice_cce_loss'
    metrics_train='MeanDice'
    metrics_valid='MeanDice'
    metrics_test='MeanDice'
    output_dir='data/results/hyperparameter_testing_test/'$JOB_ID
    mkdir -p $output_dir
    
    # Run train
    python3 train_cropdata.py \
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
	    --output_dir=$output_dir \
	    --refresh_rate=0 \
	    --schedule=$schedule \
	    --seed=$seed
}



function main(){
    let n_jobs=$n_jobs-1
    if [ $run_type == 'slurm' ] ; then
	sbatch --array=$n_start-$n_jobs%10 --output=slurm_outputs/hyperparam_test_%a.out $0 call-train
    else
	call-train 1
    fi
}



if [[ $1 ]] ; then
    command=$1
    echo $1
    shift
    $command $@
else
    main
fi
