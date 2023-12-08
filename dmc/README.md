# DNC-Medium

This codebase was adapted from [Primacy bias](https://github.com/evgenii-nikishin/rl_with_resets).

## Requirements

We assume you have access to a GPU that can run CUDA 11.1 and CUDNN 8. 
Then, the simplest way to install all required dependencies is to create an anaconda environment by running

```
conda env create -f requirements.yml
```

After the instalation ends you can activate your environment with
```
conda activate dmc
```


## Instructions

To run a single run, use the `train_pixels.py` script
```
MUJOCO_GL=egl python train_pixels.py 
```

To run on a different game, select the game as
```
MUJOCO_GL=egl python train_pixels.py --env_name quadruped-run 
```

To run the DMC-Medium benchmark (11 games with 5 random sees), use `run_parallel.py` script
```
python run_parallel.py
```

To reproduce the performance of PLASTIC, run the below script
```
bash scripts/plastic.sh
```

