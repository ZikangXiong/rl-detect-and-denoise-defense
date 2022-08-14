# Defense Adversarial Attacks in Deep Reinforcement Learning via Detecting and Denoising

Code repository of ECML-PKDD 22' paper **Defending Observation Attacks in Deep Reinforcement Learning via Detection and Denoising**

[ArXiv](https://arxiv.org/abs/2206.07188)

## Overview

<img width="850" alt="image" src="https://user-images.githubusercontent.com/73256697/184552304-f0de4cb3-1e6a-4fc1-90f5-35edaaf4417d.png">

![image](https://user-images.githubusercontent.com/73256697/184551590-e9fda43c-2344-480d-9edb-c46ee8429eb1.png)


## Structure
```
├── neural_shield
│   ├── attack          # attack algorithms
│   │   ├── common
│   │   ├── offline     # offline attacks
│   │   └── online      # online attacks
│   ├── benchmark       # robots in simulation
│   ├── config.py       # config file, including data path, simulation, attack, and defense parameters
│   ├── controller      # pre-trained controller loader
│   ├── defense         # detect-and-denoise defense
│   └── evaluation      # evaluation scripts
└── README.MD
```

## Quick Start
All the attack and defense related functions are summarized in   
`neural_shield/evaluation/run.py`  

## Bibtex
```bibtex
@inproceedings{xiong2022defending,
  title={Defending Observation Attacks in Deep Reinforcement Learning via Detection and Denoising},
  author={Xiong, Zikang and Eappen, Joe and Zhu, He and Jagannathan, Suresh},
  booktitle={2022 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  year={2022}
}
```
