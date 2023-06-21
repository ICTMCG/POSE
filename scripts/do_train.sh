
device=cuda:6
mode=POSE
config_name=progressive

python3 main.py \
--config_name $config_name \
--device $device \
--mode $mode \
--data split1

