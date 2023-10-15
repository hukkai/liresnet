# A Recipe for Improved Certifiable Robustness: Capacity and Data

[Paper](https://arxiv.org/pdf/2310.02513.pdf)

To train a model, see `run.sh` for example. You can also find [configs](/configs) for other datasets.

A detailed README and checkpoints will be updated soon. For any questions, feel free to submit an issue or email `kaihu@cmu.edu`.

Results on 4 datasets

| dataset       | clean accuracy | VRA@36/255 | VRA@72/255 | VRA@108/255 |
| ------------- | -------------- | ---------- | ---------- | ----------- |
| CIFAR-10      | 87.0           | 78.1       | 66.6       | 53.5        |
| CIFAR-100     | 62.1           | 50.1       | 38.5       | 29.0        |
| Tiny-ImageNet | 48.4           | 37.0       | 26.8       | 18.6        |
| ImageNet      | 49.0           | 38.3       | -          | -           |
