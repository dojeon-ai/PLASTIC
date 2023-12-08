# Synthetic

## Requirements

We assume you have access to a GPU that can run CUDA 11.1 and CUDNN 8. 
Then, the simplest way to install all required dependencies is to create an anaconda environment by running

```
conda env create -f requirements.yaml
```

After the instalation ends you can activate your environment with
```
conda activate synthetic
```

## Instructions

To test the baseline for input plasticity, run
```
python input_adaptation.py 
```

To test the PLASTIC for input plasticity, run
```
python input_adaptation.py --optimizer_type=sam --backbone_norm=ln --policy_reset=True --policy_crelu=True
```

To test the baseline for output plasticity, run
```
python output_adaptation.py 
```

To test the PLASTIC for output plasticity, run
```
python output_adaptation.py --optimizer_type=sam --backbone_norm=ln --policy_reset=True 
```





