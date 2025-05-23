for config in n2 ch4n2 cof zeolite heat bandgap; do
    config_path="./configs/multitask/vn/${config}.yaml"
    echo $config_path
    ./scripts/train.sh $config_path
done