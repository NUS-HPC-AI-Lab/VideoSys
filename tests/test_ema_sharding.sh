colossalai run --nproc_per_node 2 --master_port 26550 test_ema_sharding.py --model DiT-XL/2 --grad_checkpoint \
 --epoch 1
