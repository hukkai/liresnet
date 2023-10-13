import argparse

import torch
import yaml

import models


def get_args():
    parser = argparse.ArgumentParser(
        'Check the lipschitz lower bound for Globally-Robust Neural Networks')

    parser.add_argument('--config',
                        type=str,
                        help='path to the config yaml file')
    parser.add_argument('--checkpoint', default=str)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg['model']
    dataset_cfg = cfg['dataset']

    model_cfg['num_lc_iter'] = 1000
    model = models.GloroNet(**model_cfg, **dataset_cfg)

    params = torch.load(args.checkpoint, 'cpu')['backbone']
    model.load_state_dict(params)
    model.eval()

    dim = max(model.fc.weight.shape)
    model.fc.weight = torch.nn.Parameter(torch.eye(dim))
    model.fc.bias = torch.nn.Parameter(torch.zeros(dim))

    inputs = torch.rand(32, 3, args.input_size, args.input_size) - 0.5

    inputs = inputs.cuda()
    inputs.requires_grad = True
    model = model.cuda()

    subL = model.sub_lipschitz()
    print('Computed sub_lipschitz is %.2f ' % subL.item())

    optimizer = torch.optim.Adam([inputs], lr=1e-4)

    lc = []
    for k in range(10000):
        optimizer.zero_grad()
        outputs = model(inputs)
        diff = (outputs[:16] - outputs[16:]).pow(2).sum(1).sqrt()
        input_diff = (inputs[:16] - inputs[16:]).pow(2).sum((1, 2, 3)).sqrt()
        loss = diff / input_diff.clamp_min(1e-9)
        (-loss.mean()).backward()
        optimizer.step()
        lc.append(loss.max().item())
        if k % 100 == 0:
            print(loss.max().item())
    print('Computed sub_lipschitz is %.2f ' % subL)
    print('Lowe Bound of sub_lipschitz is %.2f ' % loss.max().item())
    torch.save(lc, 'curve.pth')


if __name__ == '__main__':
    main()
