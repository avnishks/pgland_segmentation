#!/bin/bash

#set -x

DATA_DIR=data
LABEL_DIR=$DATA_DIR/labels
IMAGE_DIR=$DATA_DIR/images

csv_file=data_config.csv

echo -n > $csv_file

for label_file in $LABEL_DIR/* ; do
    subject=$(basename $label_file | cut -d "." -f 1)
    flip=$(basename $label_file | cut -d "." -f 3)
    if [ $flip == "norev" ] ; then
	subject=$(basename $label_file | cut -d "." -f 1)
	image_file=$IMAGE_DIR/$subject.dc.$flip.mgz
	
	echo $image_file","$label_file >> $csv_file
    fi
done
