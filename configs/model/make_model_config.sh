#!/bin/bash

#set -x
PATH='/space/azura/1/users/kl021/Code/PituitarySegmentation/configs/model'
file_path=$PATH/model_config_

activation_functions=('ELU' 'ReLU' 'LeakyReLU')
n_layers=(3 4)
#LR_starts=(0.001 0.0005 0.0001)
#weight_decays=(0.01 0.0001 0.00001)

n=0
for activ_fn in ${activation_functions[@]} ; do
    for nL in ${n_layers[@]} ; do
	for LR in ${LR_starts[@]} ; do
	    for wdecay in ${weight_decays[@]} ; do
		file=$file_path$n.txt
		echo $n 
		echo -n > $file
		
		echo "activation_function="$activ_fn >> $file
		echo "n_layers="$nL >> $file
		echo "LR="$LR >> $file
		echo "weight_decay="$$wdecay >> $file
		let n=$n+1
	    done
	done
    done
done

