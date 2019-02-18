# Train binarized ResNets on CIFAR-10
# Use warm start
# Use Adam with const lr = 0.01 for Prox and LR Decay for BC
# Update 11/22: testing only 200 epochs training + 100 after freeze
LR=0.01
DEPTH=20

for i in 0
do
    python main_binary_reg.py --model resnet --resume results/resnet"$DEPTH" --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_lazy_Adam_run_$i --dataset cifar10 --gpu $i --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir release_test/resnet"$DEPTH"_BinaryConnect_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --binary_regime --projection_mode lazy --freeze_epoch 200 &
done

wait
echo all processes complete
