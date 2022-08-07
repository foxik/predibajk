#!/bin/sh

for d in model-s-f[0-9]; do
  qsub -q gpu-troja.q -p -110 -l gpu=1,gpu_ram=11G,mem_free=16G,h_data=20G -pe smp 4 -o $d/dev.tsv -e $d/dev.log withcuda112 ~/venvs/tf-2.8/bin/python predict_jackknife.py $d
done
