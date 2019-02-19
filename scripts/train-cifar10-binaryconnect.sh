# Train binarized ResNets on CIFAR-10 with BinaryConnect
LR=0.01
DEPTH=20

for i in 0 1 2 3
do
    python main_binary_reg.py --model resnet --resume results/resnet"$DEPTH" --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_bc_Adam_run_$i --dataset cifar10 --gpu $i --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir tb/resnet"$DEPTH"_bc_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --binary_regime --projection_mode lazy --freeze_epoch 200 &
done

wait
echo all processes complete
