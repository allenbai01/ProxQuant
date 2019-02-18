# Train binarized ResNets on CIFAR-10
# Use warm start
# Use Adam with const lr = 0.01 for Prox and LR Decay for BC
# Update 11/22: testing only 200 epochs training + 100 after freeze
LR=0.01
DEPTH=20

for i in 0
do
    python main_binary_reg.py --model resnet --resume results/resnet"$DEPTH" --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_prox_Adam_run_$i --dataset cifar10 --gpu $i --batch-size 128 --epochs 300 --reg_rate 1e-4 --tb_dir release_test/resnet"$DEPTH"_prox_Adam_Freeze_200_run_$i --optimizer Adam --lr $LR --projection_mode prox --freeze_epoch 200 &
done

# for i in 0 1 2 3
# do
#     python main_binary_reg.py --model resnet --resume results/resnet"$DEPTH"_gluon_rep"$REP" --model_config "{'depth': $DEPTH}" --save rep"$REP"/resnet"$DEPTH"_lazy_Adam_run_$i --dataset cifar10 --gpu $i --batch-size 128 --epochs 300 --binary_reg 1.0 --tb_dir tb/reps_short/resnet"$DEPTH"_lazy_Adam_Freeze_200_rep_"$REP"_run_$i --optimizer Adam --lr $LR --binary_regime --projection_mode lazy --freeze_epoch 200 &
# done

wait
echo all processes complete
