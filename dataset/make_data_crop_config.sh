#!/bin/bash

set -x

DATA_DIR=data
LABEL_DIR=$DATA_DIR/labels
IMAGE_DIR=$DATA_DIR/images
LABEL_DIR_crop=${DATA_DIR}_cropped/labels
IMAGE_DIR_crop=${DATA_DIR}_cropped/images
mkdir -p $LABEL_DIR_crop $IMAGE_DIR_crop

csv_file=data_config_crop.csv

echo -n > $csv_file

for label_file in $LABEL_DIR/* ; do
    subject=$(basename $label_file | cut -d "." -f 1)
    flip=$(basename $label_file | cut -d "." -f 3)
    if [ $flip == "norev" ] ; then
	subject=$(basename $label_file | cut -d "." -f 1)
	label_file_crop=$LABEL_DIR_crop/$(basename $label_file)
	image_file_crop=$IMAGE_DIR_crop/$subject.dc.$flip.mgz
	
	echo $image_file_crop","$label_file_crop >> $csv_file
    fi
done
