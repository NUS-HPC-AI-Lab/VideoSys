#torchrun --nnodes=1 --nproc_per_node=2 --master-port=29502 train.py --model DiT-XL/2 --grad_checkpoint --outputs /data/personal/nus-lzm/dit_output \
#--load /data/personal/nus-lzm/dit_output/000-DiT-XL-2/epoch0-step10
colossalai run --nproc_per_node 2 --master_port 26550 train.py --model DiT-XL/2 --grad_checkpoint --outputs /data/personal/nus-lzm/dit_output \
--load /data/personal/nus-lzm/dit_output/011-DiT-XL-2/epoch0-step20
