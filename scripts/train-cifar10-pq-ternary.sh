# Train ternarized ResNets on CIFAR-10
DEPTH=20

for i in 0 1 2 3
do
    python main_binary_reg.py --resume results/resnet"$DEPTH" --model resnet --model_config "{'depth': $DEPTH}" --dataset cifar10 --gpu $i --epochs 600 --reg_rate 1e-4 --tb_dir tb/resnet"$DEPTH"_Adam_Freeze_400_run_"$i" --optimizer Adam --lr 0.01 --projection_mode prox_ternary --freeze_epoch 400 &
done

wait
echo all processes complete
