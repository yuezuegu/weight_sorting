#!/bin/bash

dirname='/home/acy/Downloads/ILSVRC2012_img_val'
l=$(ls $dirname | shuf -n 5) 

for i in ${l[@]}
do
	python main.py /home/acy/Downloads/ILSVRC2012_img_val/ $i &
done


