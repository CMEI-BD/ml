#!/bin/sh

file="std_sogou"
for i in $(seq -w 00 31); do
	file_path=$file$i
	log_path="$file$i.log"
	nohup python jieba_cut.py $file_path > $log_path &
done
