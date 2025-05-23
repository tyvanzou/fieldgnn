for grid_size in `seq 2 2 12`; do
    config_path="./configs/ablation/grid/schnet.grid${grid_size}.yaml"
    echo $config_path
    ./scripts/train.sh ${config_path}
done
