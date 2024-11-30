import argparse
import os
import time

import timm.optim
import torch
import yaml

import models
import tools


def get_args():
    parser = argparse.ArgumentParser(
        'Training Globally-Robust Neural Networks')

    parser.add_argument('--config',
                        type=str,
                        help='path to the config yaml file')
    # checkpoint saving
    parser.add_argument('--work_dir', default='./checkpoint/', type=str)
    parser.add_argument('--ckpt_prefix', default='', type=str)
    parser.add_argument('--max_save', default=2, type=int)
    parser.add_argument('--resume_from', default='', type=str)
    # distributed training
    parser.add_argument('--launcher',
                        default='slurm',
                        type=str,
                        help='should be either `slurm` or `pytorch`')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg['model']
    train_cfg = cfg['training']
    dataset_cfg = cfg['dataset']
    gloro_cfg = cfg['gloro']

    if args.ckpt_prefix == '':
        depth, width = model_cfg['depth'], model_cfg['width']
        prefix = f"{dataset_cfg['name']}-{depth}x{width}"
        args.ckpt_prefix = prefix

    if args.resume_from:
        ckpt = torch.load(args.resume_from, 'cpu')
        backbone_ckpt = ckpt['backbone']
        optimizer_ckpt = ckpt['optimizer']
        start_epoch = ckpt['start_epoch']
        current_iter = ckpt['current_iter']
        training_logs = ckpt['training_logs']
        resume = True
    else:
        start_epoch = 0
        training_logs = []
        resume = False

    rank, local_rank, num_gpus = tools.init_DDP(args.launcher)
    print('Inited distributed training!')

    if local_rank == 0:
        os.system(f'cat {args.config}')

    print(f'Use checkpoint prefix: {args.ckpt_prefix}')

    train_loader, train_sampler, val_loader, _ = tools.data_loader(
        data_name=dataset_cfg['name'],
        batch_size=train_cfg['batch_size'] // num_gpus,
        num_classes=dataset_cfg['num_classes'],
        seed=dataset_cfg.get('seed', 2023))  # if seed is not given, use 2023

    aug_loader, aug_sampler, _, _ = tools.data_loader(
        data_name='ddpm',
        batch_size=train_cfg['batch_size'] // num_gpus * 3,
        num_classes=dataset_cfg['num_classes'],
        seed=dataset_cfg.get('seed', 2023))
    aug_iter = iter(aug_loader)

    model = models.GloroNet(**model_cfg, **dataset_cfg)
    if resume:
        model.load_state_dict(backbone_ckpt)
    print(model)
    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

    if cfg['training']['nadam']:
        optim_fn = torch.optim.NAdam
    else:
        optim_fn = torch.optim.Adam
    optimizer = optim_fn(model.parameters(),
                         lr=train_cfg['lr'],
                         weight_decay=train_cfg['weight_decay'])
    if cfg['training']['lookahead']:
        optimizer = timm.optim.Lookahead(optimizer)

    scheduler = tools.lr_scheduler(iter_per_epoch=len(train_loader),
                                   max_epoch=train_cfg['epochs'],
                                   warmup_epoch=train_cfg['warmup_epochs'])

    if resume:
        optimizer.load_state_dict(optimizer_ckpt)
        scheduler.current_iter = current_iter
        scheduler.base_lr = optimizer_ckpt['param_groups'][0]['initial_lr']
        sub_lipschitz = model.module.sub_lipschitz().item()

    def eps_fn(epoch):
        ratio = min(epoch / train_cfg['epochs'] * 2, 1)
        ratio = gloro_cfg['min_eps'] + (gloro_cfg['max_eps'] -
                                        gloro_cfg['min_eps']) * ratio
        return gloro_cfg['eps'] * ratio

    os.makedirs(args.work_dir, exist_ok=True)

    train_fn = getattr(models, gloro_cfg['loss_type'])

    print('Begin Training')
    for log in training_logs:
        print(log)
    t = time.time()
    for epoch in range(start_epoch, train_cfg['epochs']):
        eps = eps_fn(epoch)
        train_sampler.set_epoch(epoch)
        # aug_sampler.set_epoch(epoch)
        model.module.set_num_lc_iter(model_cfg['num_lc_iter'])

        model.train()
        correct_vra = correct = total = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            bs = inputs.shape[0]
            sub_lipschitz = model.module.sub_lipschitz()

            try:
                input2, target2 = next(aug_iter)
            except StopIteration:
                aug_sampler.set_epoch(epoch)
                aug_iter = iter(aug_loader)
                input2, target2 = next(aug_iter)

            inputs = torch.cat([inputs, input2])
            targets = torch.cat([targets, target2])

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            y, y_, loss = train_fn(model,
                                   x=inputs,
                                   label=targets,
                                   lc=sub_lipschitz,
                                   eps=eps,
                                   return_loss=True)

            _ = scheduler.step(optimizer)
            loss.backward()
            if train_cfg['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_cfg['grad_clip_val'])

            optimizer.step()

            correct += y.argmax(1).eq(targets)[:bs].sum().item()
            correct_vra += y_.argmax(1).eq(targets)[:bs].sum().item()
            total += bs

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        if epoch % 5 == 0 or epoch > train_cfg['epochs'] * 0.9:
            model.eval()
            model.module.set_num_lc_iter(500)  # let the power method converge
            # only need to comput the sub_lipschitz only once for validation
            sub_lipschitz = 1.0
            if gloro_cfg['eps'] != 0:
                sub_lipschitz = model.module.sub_lipschitz().item()

            val_correct_vra = val_correct = val_total = 0.

            for inputs, targets in val_loader:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                with torch.no_grad():
                    y, y_, _ = models.trades_loss(model,
                                                  x=inputs,
                                                  label=targets,
                                                  eps=gloro_cfg['eps'],
                                                  lc=sub_lipschitz,
                                                  return_loss=False)

                val_correct += y.argmax(1).eq(targets).sum().item()
                val_correct_vra += y_.argmax(1).eq(targets).sum().item()
                val_total += targets.size(0)

            collect_info = [
                correct_vra, correct, total, val_correct_vra, val_correct,
                val_total
            ]
            collect_info = torch.tensor(collect_info,
                                        dtype=torch.float32,
                                        device=inputs.device).clamp_min(1e-9)
            torch.distributed.all_reduce(collect_info)

            acc_train = 100. * collect_info[1] / collect_info[2]
            acc_val = 100. * collect_info[4] / collect_info[5]

            acc_vra_train = 100. * collect_info[0] / collect_info[2]
            acc_vra_val = 100. * collect_info[3] / collect_info[5]
        else:
            acc_train = acc_val = acc_vra_train = acc_vra_val = 0.0

        used = time.time() - t
        t = time.time()

        string = (f'Epoch {epoch}: '
                  f'Train acc{acc_train: .2f}%,{acc_vra_train: .2f}%; '
                  f'val acc{acc_val: .2f}%,{acc_vra_val: .2f}%. '
                  f'sub_lipschitz:{sub_lipschitz: .2f}. '
                  f'Time:{used / 60: .2f} mins.')

        print(string)
        training_logs.append(string)
        if rank == 0:
            state = dict(backbone=model.module.state_dict(),
                         optimizer=optimizer.state_dict(),
                         start_epoch=epoch + 1,
                         current_iter=scheduler.current_iter,
                         training_logs=training_logs,
                         configs=cfg)

            try:
                path = f'{args.work_dir}/{args.ckpt_prefix}_{epoch}.pth'
                torch.save(state, path)
            except PermissionError:
                print('Error saving checkpoint!')
                pass
            if epoch >= args.max_save:
                path = (f'{args.work_dir}/'
                        f'{args.ckpt_prefix}_{epoch - args.max_save}.pth')
                os.system('rm -f ' + path)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
