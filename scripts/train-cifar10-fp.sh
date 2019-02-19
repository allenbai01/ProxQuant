# Train a Full-Precision ResNet
DEPTH=20
python main_binary_reg.py --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH" --dataset cifar10 --batch-size 128 --gpu 0 --epochs 200 --tb_dir tb/ResNet"$DEPTH"_FP
