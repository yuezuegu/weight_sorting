#!/bin/bash

rm log_noverbose.txt
rm log.txt

source /home/acy/tensorflow/bin/activate

dirname='/home/acy/Downloads/ILSVRC2012_img_val/'
l=$(ls $dirname | shuf -n 5) 

for i in ${l[@]}
do
	python main.py /home/acy/Downloads/ILSVRC2012_img_val/ $i |& tee -a log.txt
done


