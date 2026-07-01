import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from test_ensemble import (
    MLETestDataset,
    run_st_split,
    run_mt_split,
)


SINGLE_TASK = 'SigLIP2_384_Image'
MULTI_TASK = 'SigLIP2_ViTG_384_Image'
MODEL_CHOICES = (SINGLE_TASK, MULTI_TASK)


MODEL_CONFIG = {
    SINGLE_TASK: {
        'crop': 384,
        'run_fn': run_st_split,
        'label': 'SigLIP2_384_Image (single-task)',
    },
    MULTI_TASK: {
        'crop': 378,
        'run_fn': run_mt_split,
        'label': 'SigLIP2_ViTG_384_Image_MT (multi-task)',
    },
}


@torch.no_grad()
def run_single(pth_path, mat_path, loader, device, run_fn):
    print('Loading checkpoint: %s' % pth_path)
    print('Loading mapping mat: %s' % mat_path)
    return run_fn(pth_path, mat_path, loader, device)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = MODEL_CONFIG[args.model]

    pth_path = args.ckpt_path
    if not os.path.isfile(pth_path):
        raise FileNotFoundError('Checkpoint not found: %s' % pth_path)

    mat_path = args.mat_path
    if mat_path is None:
        # default: same stem as the .pth, with .mat extension
        mat_path = os.path.splitext(pth_path)[0] + '.mat'
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(
            'Mapping .mat not found: %s (pass --mat_path explicitly)' % mat_path)

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    tf = transforms.Compose([
        transforms.Resize(432),
        transforms.CenterCrop(cfg['crop']),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    ds = MLETestDataset(args.test_dir, args.metadata_csv, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    canonical_ids = [sid for sid, _ in ds.items]
    print('Model: %s' % cfg['label'])
    print('Test images: %d' % len(ds))

    print('\n' + '=' * 60)
    print('  Single checkpoint inference')
    print('=' * 60)
    preds = run_single(pth_path, mat_path, loader, device, cfg['run_fn'])

    print('\n' + '=' * 60)
    print('  Samples: %d' % len(canonical_ids))
    print('=' * 60)

    results = []
    for sid in canonical_ids:
        mos = preds.get(sid, float('nan'))
        if args.clamp_to_range:
            mos = max(args.min_mos, min(args.max_mos, mos))
        if args.round is not None:
            mos = round(mos, args.round)
        results.append({'id': sid, 'mos': mos})

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n[Output] wrote %d samples to %s' % (len(results), args.output_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MLE single checkpoint test (one trained model)')

    parser.add_argument('--model', type=str, default=SINGLE_TASK,
                        choices=MODEL_CHOICES,
                        help='which architecture to run: SigLIP2_384_Image (single-task) or SigLIP2_ViTG_384_Image (multi-task)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='path to a single trained .pth checkpoint')
    parser.add_argument('--mat_path', type=str, default=None,
                        help='path to the .mat file for logistic mapping '
                             '(default: same stem as --ckpt_path with .mat)')

    parser.add_argument('--test_dir', type=str, default='test_images')
    parser.add_argument('--metadata_csv', type=str,
                        default='test_images/metadata.csv')
    parser.add_argument('--output_json', type=str, default='result_single.json')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--round', type=int, default=4)
    parser.add_argument('--clamp_to_range', action='store_true')
    parser.add_argument('--min_mos', type=float, default=0.0)
    parser.add_argument('--max_mos', type=float, default=10.0)

    main(parser.parse_args())
