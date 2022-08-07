#!/bin/sh

for size in m s; do
  case $size in
    s) batch=8; gpu_ram=16G;;
    m) batch=4; gpu_ram=24G;;
  esac
  for fold in "" $(seq 0 9); do
    outdir=model-$size${fold:+-f$fold}
    [ -d $outdir ] && continue
    qsub -q gpu-ms.q -l gpu=1,gpu_ram=$gpu_ram,mem_free=16G,h_data=20G -pe smp 4 -j y withcuda112 ~/venvs/tf-2.8/bin/python3 model_train.py --batch_size=$batch --efficientnetv2_size=$size ${fold:+--data_folds=10 --data_fold=$fold} --output=$outdir
  done
done
