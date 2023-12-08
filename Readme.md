# Improving Input & Label Plasticity for Sample Efficient RL

This repository is an official PyTorch implementation of the paper, PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning, NeurIPS 2023.

[arXiv](https://arxiv.org/abs/2306.10711) /
[slide](https://drive.google.com/file/d/1-QeWhom9l7mUt3m7zJV-_DIGMtL7F2Cq/view) /
[poster](https://drive.google.com/file/d/1-OTP_-rw2x-csjsJ9jH7utuHw9zDxsJc/view)

Authors: 
[Hojoon Lee](https://joonleesky.github.io/about/)\*,
[Hanseul Cho](https://hanseuljo.github.io/)\*,
[Hyunseung Kim](https://mynsng.github.io/)\*, 
Daehoon Gwak, 
Joonkee Kim, 
[Jaegul Choo](https://sites.google.com/site/jaegulchoo/), 
[Se-Young Yun](https://fbsqkd.github.io/), and 
[Chulhee Yun](https://chulheeyun.github.io/).


## Instructions

To run the synthetic experiments, please follow the instructions of `Readme.md` in the folder `synthetic`.

To run the Atari-100k experiments, please follow the instructions of `Readme.md` in the folder `atari`.

To run the DMC-M experiments, please follow the instructions of `Readme.md` in the folder `dmc`.

## Results

![plot](results.png)

- **Left:** Performance of Synthetic Experiments. Layer Normalization (LN) and SharpenessAware Minimization (SAM) considerably enhance input plasticity, while their effect on label plasticity
is marginal. Conversely, Concatenated ReLU (CReLU) and periodic reinitialization (Reset) predominantly improve label plasticity with subtle benefits on input plasticity. 

- **Right:** Performance of RL Benchmarks. PLASTIC consistently outperforms individual methods, highlighting the synergistic benefits of its integrated approach.



## Citations

```
@inproceedings{lee2023plastic,
  title={PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning},
  author={Lee, Hojoon and Cho, Hanseul and Kim, Hyunseung and Gwak, Daehoon and Kim, Joonkee and Choo, Jaegul and Yun, Se-Young and Yun, Chulhee},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Contact

For personal communication, please contact Hojoon Lee, Hanseul Cho, or Hyunseung Kim at 

`{joonleesky, jhs4015, mynsng}@kaist.ac.kr`.
