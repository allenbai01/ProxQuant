# ProxQuant: Quantized Neural Networks via Proximal Operators
This repository provides the implementation for the paper [ProxQuant: Quantized Neural Networks via Proximal Operators](https://openreview.net/forum?id=HyzMyhCcK7) by Yu Bai, Yu-Xiang Wang, and Edo Liberty. 

Our algorithm uses a suitable proximal operator (with eager projection) to perform quantization in between gradient steps. (In contrast, the straight-through gradient method is equivalent to using a hard quantization mapping with lazy projection.) The proximal operator can be customized to binary, ternary, and multi-bit quantization. 

## Requirements
1. Python 3.6
1. PyTorch (==0.4.1)
2. [TensorboardX](https://github.com/lanpa/tensorboardX)

## Train a quantized ResNet on CIFAR-10
Begin by training a full-precision ResNet-20 (similarly for depth {32, 44, 56}) as the warm start:
```
python main_binary_reg.py --model resnet --model_config "{'depth': 20}" --save resnet20 --dataset cifar10 --batch-size 128 --gpu 0 --epochs 200 --tb_dir tb/ResNet20_FP
```
Train a binarized net via ProxQuant-binary:
```
python main_binary_reg.py --model resnet --resume results/resnet20 --model_config "{'depth': 20}" --save resnet20_pq_binary_Adam_run_0 --dataset cifar10 --gpu 0 --batch-size 128 --epochs 300 --reg_rate 1e-4 --tb_dir tb/resnet20_pq_binary_Adam_Freeze_200_run_0 --optimizer Adam --lr 0.01 --projection_mode prox --freeze_epoch 200
```
Train a binarized net via BinaryConnect:
```
python main_binary_reg.py --model resnet --resume results/resnet20 --model_config "{'depth': 20}" --save resnet20_bc_Adam_run_0 --dataset cifar10 --gpu 0 --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir tb/resnet20_bc_Adam_Freeze_200_run_0 --optimizer Adam --lr 0.01 --binary_regime --projection_mode lazy --freeze_epoch 200
```
Train a ternarized net via ProxQuant-ternary:
```
python main_binary_reg.py --resume results/resnet20 --model resnet --model_config "{'depth': 20}" --dataset cifar10 --gpu $i --epochs 600 --reg_rate 1e-4 --tb_dir tb/resnet20_Adam_Freeze_400_run_"$i" --optimizer Adam --lr 0.01 --projection_mode prox_ternary --freeze_epoch 400
```

Corresponding shell scripts for parallellizing multiple runs (requires multiple GPUs) can be found in `scripts/`.

## Train a quantized LSTM on Penn Treebank
TBA.

## Miscellanous
The code is based on [BinaryNet.pytorch](https://github.com/itayhubara/BinaryNet.pytorch).
