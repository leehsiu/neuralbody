This is an unofficial implementation for [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://arxiv.org/pdf/2012.15838.pdf), which uses a combination of a vertex-based representation and NeRF to handle dynamics. There are some interesting points in this paper, while also a lot of issues to be fixed and improved. This repo removes some redundancy and provides better readability(in mho), to push forward related researches.

## Installation

Please see [docker/README.md](docker/README.md).


## Run the code

see [run.sh](run.sh).

The dataset can be accessed through the official repo [NeuralBody](https://github.com/zju3dv/neuralbody/)


## References
```
@inproceedings{peng2021neural,
  title={Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans},
  author={Peng, Sida and Zhang, Yuanqing and Xu, Yinghao and Wang, Qianqian and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2021}
}
```
