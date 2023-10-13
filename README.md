# A Recipe for Improved Certifiable Robustness: Capacity and Data

To train a model, run:

```
#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 \
    --master_port $MASTER_PORT train.py --launcher=pytorch \
    --config='configs/cifar10.yaml'
```

You can also find [configs](\configs) for other datasets.

A detailed README and checkpoints will be updated soon.
