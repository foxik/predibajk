#!/bin/sh

for d in model-s; do
  qsub -q gpu-troja.q -p -110 -l gpu=1,gpu_ram=11G,mem_free=16G,h_data=20G -pe smp 4 -o $d/all.tsv -e $d/all.log withcuda112 ~/venvs/tf-2.8/bin/python predict.py $d images/1[5-9]*.jpg images/[2-9]*.jpg
done
