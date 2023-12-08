import subprocess
import argparse
import json
import copy
import wandb
import itertools
import os
import multiprocessing as mp
import multiprocessing
import numpy as np
import time

games_dict = { 
'medium': [
    'walker-run',# 2
    'acrobot-swingup',# 2
    'cartpole-swingup_sparse',# 8
    'cheetah-run',# 4
    'finger-turn_easy',# 2
    'finger-turn_hard',# 2
    'hopper-hop',# 2
    'quadruped-run',# 2
    'quadruped-walk',# 2
    'reacher-easy',# 4
    'reacher-hard',# 4
]
}


def run_script(script_name):
    print(script_name)
    subprocess.run(script_name, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--group_name',   type=str,     default='test')
    parser.add_argument('--exp_name',     type=str,     default='test')
    parser.add_argument('--num_seeds',    type=int,     default=5)
    parser.add_argument('--num_devices',  type=int,     default=3)
    parser.add_argument('--num_exp_per_device',  type=int,  default=2)
    parser.add_argument('--gpu_indices',  nargs="+",  type=int)
    parser.add_argument('--use_sam', type=str, default='True') 
    parser.add_argument('--resets', type=str, default='True')
    parser.add_argument('--use_LN', type=str, default='True') 
    parser.add_argument('--use_CReLU', type=str, default='True')

    args = vars(parser.parse_args())
    seeds = np.arange(args.pop('num_seeds'))

    gpu_indices = args.pop('gpu_indices')
    num_devices = args.pop('num_devices')
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 
    
    games = games_dict['medium']
    # create configurations for child run
    experiments = []
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(args)
        exp['seed'] = seed
        exp['env_name'] = str(game)

        experiments.append(exp)
        print(exp)

    # run parallell experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method('spawn') 
    available_gpus = list(range(num_devices))
    process_dict = {gpu_id: [] for gpu_id in available_gpus}

    for exp in experiments:
        wait = True
        while wait:
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)
            
            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False    
                    break
            
            time.sleep(1)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        cmd = 'CUDA_VISIBLE_DEVICES={} XLA_PYTHON_CLIENT_PREALLOCATE=False EGL_DEVICE_ID={} MUJOCO_GL=egl \
               python train_pixels.py \
                --env_name {} \
                --seed {} \
                --exp_name {} \
                --use_sam={} \
                --resets={} \
                --use_CReLU={} \
                --use_LN={}'.format(
                    str(gpu_indices[gpu_id]), 
                    str(gpu_indices[gpu_id]), 
                    exp['env_name'],
                    exp['seed'], 
                    exp['exp_name'],
                    exp['use_sam'],
                    exp['resets'],
                    exp['use_CReLU'],
                    exp['use_LN']
                )
        
        process = multiprocessing.Process(target=run_script, args=(cmd,))
        process.start()
        processes.append(process)

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)
