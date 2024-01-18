#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=1-04:00:00
#SBATCH --job-name=pseg_config_optimizer_2

WDIR=/space/azura/1/users/kl021/Code/PituitarySegmentation
n_jobs=$(ls $WDIR/configs/optimizer/optimizer_config_*.txt | wc -l)

set -x

#run_type=slurm
run_type=debug
test_job_id=45

n_workers=4
n_layers=3

n_start=45
let n_jobs=$n_jobs-1


function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
    else
	n_workers=8
        JOB_ID=$1
    fi
    
    # Get config parameters from file
    config_file=$WDIR/configs/optimizer/optimizer_config_$JOB_ID.txt
    lr_start=$(sed '1q;d' $config_file | cut -d '=' -f 2)
    weight_decay=$(sed '2q;d' $config_file | cut -d '=' -f 2)
    lr_scheduler=$(sed '3q;d' $config_file | cut -d '=' -f 2)

    # Define non-default inputs
    data_config='configs/data/data_config_t1.csv'
    max_n_epochs=1500
    metrics_train="MeanDice"
    metrics_valid="MeanDice"
    metrics_test="MeanDice"

    output_dir=$WDIR/data/results/optimizer_config/$JOB_ID
    mkdir -p $output_dir
    
    # Run train
    python3 train.py \
	    --data_config $data_config \
	    --weight_decay $weight_decay \
	    --lr_start $lr_start \
	    --max_n_epochs $max_n_epochs \
	    --metrics_test $metrics_test \
	    --metrics_train $metrics_train \
	    --metrics_valid $metrics_valid \
	    --n_workers $n_workers \
	    --optim "Adam" \
	    #--output_dir $output_dir \
	    --lr_scheduler $lr_scheduler
}



function main(){
    if [ $run_type == 'slurm' ] ; then
	sbatch --array=$n_start-$n_jobs --output=slurm_outputs/optimizer_config/%a.out $0 call-train
    else
	call-train $test_job_id
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
