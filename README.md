# 🚀 LiResNet: An Network Architecture for Training Certifiable Robust Models


This repository provides the implementation of our cutting-edge research on certifiable robust models. We proudly present the LiResNet model introduced at NeurIPS 2023 and its subsequent improvements now available as a preprint on arXiv. Our works are based on the training and certification schemes in [GloRo Nets](https://arxiv.org/abs/2102.08452). 

- [NeurIPS 2023] [Unlocking Deterministic Robustness Certification on ImageNet](https://arxiv.org/abs/2301.12549)
- [Preprint] [A Recipe for Improved Certifiable Robustness: Capacity and Data](https://arxiv.org/abs/2310.02513)
  - [Direct Paper Download](https://arxiv.org/pdf/2310.02513.pdf)


## 🚀 Getting Started:
- For training a model, check out our `run.sh` as a starting point.
- Dive into our [configs](/configs) for additional dataset configurations.

## 🔜 Upcoming Updates:
- A comprehensive README.
- Pre-trained model checkpoints.

## 📈 Main Results:
| dataset       | clean accuracy | VRA@36/255 | VRA@72/255 | VRA@108/255 |
|:-------------:|:--------------:|:----------:|:----------:|:-----------:|
| CIFAR-10      | 87.0%          | 78.1%      | 66.6%      | 53.5%       |
| CIFAR-100     | 62.1%          | 50.1%      | 38.5%      | 29.0%       |
| Tiny-ImageNet | 48.4%          | 37.0%      | 26.8%      | 18.6%       |
| ImageNet      | 49.0%          | 38.3%      | -          | -           |

## 🤝 Support:
- Encountering issues? Submit an Issue.
- For specific inquiries, 📧 drop an email to `kaihu@cmu.edu`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations
If you find this repository useful, consider to use the following citations

```
@INPROCEEDINGS{hu2023scaling,
    title={Unlocking Deterministic Robustness Certification on ImageNet},
    author={Kai Hu and Andy Zou and Zifan Wang and Klas Leino and Matt Fredrikson},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=SHyVaWGTO4}
}

@misc{hu2023recipe,
    title={A Recipe for Improved Certifiable Robustness: Capacity and Data}, 
    author={Kai Hu and Klas Leino and Zifan Wang and Matt Fredrikson},
    year={2023},
    eprint={2310.02513},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@INPROCEEDINGS{leino21gloro,
    title = {Globally-Robust Neural Networks},
    author = {Klas Leino and Zifan Wang and Matt Fredrikson},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2021}
}
```