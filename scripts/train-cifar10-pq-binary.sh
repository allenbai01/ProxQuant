# Train binarized ResNets on CIFAR-10
LR=0.01
DEPTH=20

for i in 0 1 2 3
do
    python main_binary_reg.py --model resnet --resume results/resnet"$DEPTH" --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_prox_Adam_run_$i --dataset cifar10 --gpu $i --batch-size 128 --epochs 300 --reg_rate 1e-4 --tb_dir tb/resnet"$DEPTH"_prox_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --projection_mode prox --freeze_epoch 200 &
done

wait
echo all processes complete
