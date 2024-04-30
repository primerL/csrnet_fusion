

log_name="multi"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=2

CUDA_VISIBLE_DEVICES=4 python /root/ai-fdu.github.io/pj3/train.py \
--method 2 \
> /root/ai-fdu.github.io/pj3/result/$log_name.txt 2>&1 & echo $! > /root/ai-fdu.github.io/pj3/result/pid_${log_name}.txt 