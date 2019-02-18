# Train ternarized ResNets on CIFAR-10
# Use either cold or warm start, freeze binarization at epoch 400
# Use Adam with const lr = 0.01
DEPTH=20


for i in 0
do
    python main_binary_reg.py --resume results/resnet"$DEPTH" --model resnet --model_config "{'depth': $DEPTH}" --dataset cifar10 --gpu $i --epochs 600 --reg_rate 1e-4 --tb_dir release_test/resnet"$DEPTH"_Adam_Freeze_400_run_"$i" --optimizer Adam --lr 0.01 --projection_mode prox_ternary --freeze_epoch 400 &
done

wait
echo all processes complete
