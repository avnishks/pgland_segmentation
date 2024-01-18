#!/bin/bash

#set -x
PATH='/space/azura/1/users/kl021/Code/PituitarySegmentation/configs/optimizer'

LR_starts=(0.0001 0.0005)
weight_decays=(0.001 0.0025 0.005)
scheduler=(StepLR PolynomialLR ExponentialLR)

n=45
for LR in ${LR_starts[@]} ; do
    for wdecay in ${weight_decays[@]} ; do
	for sched in ${scheduler[@]} ; do
	    file=$PATH/optimizer_config_$n.txt
	    echo $n $LR $wdecay $sched
	    echo -n > $file
	    
	    echo "LR="$LR >> $file
	    echo "weight_decay="$wdecay >> $file
	    echo "scheduler="$sched >> $file
	    let n=$n+1
	done
    done
done

