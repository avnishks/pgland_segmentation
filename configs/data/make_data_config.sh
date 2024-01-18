#!/bin/bash

#set -x

DATA_DIR=data/input
csv_t1=dataset/data_config_t1.csv
csv_t2=dataset/data_config_t2.csv
csv_qt1=dataset/data_config_qt1.csv
csv_pd=dataset/data_config_pd.csv

csv_t1t2=dataset/data_config_t1t2.csv
csv_t1qt1=dataset/data_config_t1qt1.csv
csv_t1pd=dataset/data_config_t1pd.csv

csv_t1t2pd=dataset/data_config_t1t2pd.csv
csv_t1t2qt1=dataset/data_config_t1t2qt1.csv
csv_t1pdqt1=dataset/data_config_t1pdqt1.csv
csv_t2pdqt1=dataset/data_config_t2pdqt1.csv
csv_all=dataset/data_config_all.csv

find dataset/data_config_*.csv -print0 | while read -d $'\0' file
do
    echo -n > $file
done

for label in $DATA_DIR/labels/* ; do
    subject=$(basename $label | cut -d "." -f 1)
    t1=$DATA_DIR/intensity/$subject.intensity.mgz
    t2=$DATA_DIR/t2/$subject.t2.mgz
    qt1=$DATA_DIR/qt1/$subject.qt1.mgz
    pd=$DATA_DIR/pd/$subject.pd.mgz
    
    echo $t1","$label >> $csv_t1
    echo $t2","$label >> $csv_t2
    echo $qt1","$label >> $csv_qt1
    echo $pd","$label >> $csv_pd

    echo $t1","$t2","$label >> $csv_t1t2
    echo $t1","$qt1","$label >> $csv_t1qt1
    echo $t1","$pd","$label >> $csv_t1pd

    echo $t1","$t2","$pd","$label >> $csv_t1t2pd
    echo $t1","$t2","$qt1","$label >> $csv_t1t2qt1
    echo $t1","$pd","$qt1","$label >> $csv_t1pdqt1
    echo $t2","$pd","$qt1","$label >> $csv_t2pdqt1
    echo $t1","$t2","$qt1","$pd","$label >> $csv_all
done
