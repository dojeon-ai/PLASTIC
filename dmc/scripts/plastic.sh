cd ..
python run_parallel.py \
    --group_name plastic \
    --exp_name plastic \
    --num_seeds 10 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --gpu_indices 0 1 2 3 \