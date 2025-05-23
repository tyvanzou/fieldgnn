for config in 0 0_2 0_4 0_6 0_8 1 1_2; do
    config_path="./configs/ablation/repulsion/schnet.repul${config}.yaml"
    echo $config_path
    ./scripts/train.sh $config_path
done