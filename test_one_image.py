import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from test_ensemble import (
    SigLIP2_384_Image,
    SigLIP2_ViTG_384_Image_MT,
    load_checkpoint_state_dict,
    logistic_func,
    fit_mapping_params_st,
    fit_mapping_params_mt,
)


SINGLE_TASK = 'st'
MULTI_TASK = 'mt'
MODEL_CHOICES = (SINGLE_TASK, MULTI_TASK)


MODEL_CONFIG = {
    SINGLE_TASK: {
        'crop': 384,
        'build_fn': lambda: SigLIP2_384_Image(),
        'fit_fn': fit_mapping_params_st,
        'label': 'SigLIP2_384_Image (single-task)',
    },
    MULTI_TASK: {
        'crop': 378,
        'build_fn': lambda: SigLIP2_ViTG_384_Image_MT(),
        'fit_fn': fit_mapping_params_mt,
        'label': 'SigLIP2_ViTG_384_Image_MT (multi-task)',
    },
}


@torch.no_grad()
def predict_one(image_path, pth_path, mat_path, model_type, device,
                clamp_to_range=False, min_mos=0.0, max_mos=10.0,
                round_digits=4):
    cfg = MODEL_CONFIG[model_type]

    if not os.path.isfile(image_path):
        raise FileNotFoundError('Image not found: %s' % image_path)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError('Checkpoint not found: %s' % pth_path)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError('Mapping .mat not found: %s' % mat_path)

    # model
    model = cfg['build_fn']()
    sd = load_checkpoint_state_dict(pth_path, device)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    # logistic mapping params from training-set predictions
    mapping_popt = cfg['fit_fn'](mat_path)

    # preprocess
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    tf = transforms.Compose([
        transforms.Resize(432),
        transforms.CenterCrop(cfg['crop']),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    image = Image.open(image_path).convert('RGB')
    image = tf(image).unsqueeze(0).to(device)

    # inference
    outputs = model(image)
    if model_type == MULTI_TASK:
        raw = float(outputs['mos'].detach().cpu().item())
        attrs = {k: float(v.detach().cpu().item()) for k, v in outputs.items()
                 if k != 'mos'}
    else:
        raw = float(outputs.detach().cpu().item())
        attrs = None

    # logistic mapping to MOS scale
    mos = float(logistic_func(np.array([raw]), *mapping_popt)[0])
    if clamp_to_range:
        mos = max(min_mos, min(max_mos, mos))
    if round_digits is not None:
        mos = round(mos, round_digits)
    return mos, raw, attrs


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mat_path = args.mat_path
    if mat_path is None:
        mat_path = os.path.splitext(args.ckpt_path)[0] + '.mat'

    cfg = MODEL_CONFIG[args.model]
    print('Model: %s' % cfg['label'])
    print('Image: %s' % args.image_path)
    print('Checkpoint: %s' % args.ckpt_path)
    print('Mat: %s' % mat_path)
    print('Device: %s' % device)
    print('-' * 40)

    mos, raw, attrs = predict_one(
        args.image_path, args.ckpt_path, mat_path, args.model, device,
        clamp_to_range=args.clamp_to_range,
        min_mos=args.min_mos, max_mos=args.max_mos,
        round_digits=args.round)

    print('Raw output: %.6f' % raw)
    print('MOS: %s' % mos)
    if attrs:
        print('Attribute predictions:')
        for name, val in attrs.items():
            if args.round is not None:
                val = round(val, args.round)
            print('  %-18s %s' % (name, val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MLE single image inference (one trained model)')

    parser.add_argument('--image_path', type=str, required=True,
                        help='path to a single image')
    parser.add_argument('--model', type=str, default=SINGLE_TASK,
                        choices=MODEL_CHOICES,
                        help='which architecture to run: st (single-task) or mt (multi-task)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='path to a single trained .pth checkpoint')
    parser.add_argument('--mat_path', type=str, default=None,
                        help='path to the .mat file for logistic mapping '
                             '(default: same stem as --ckpt_path with .mat)')

    parser.add_argument('--round', type=int, default=4)
    parser.add_argument('--clamp_to_range', action='store_true')
    parser.add_argument('--min_mos', type=float, default=0.0)
    parser.add_argument('--max_mos', type=float, default=10.0)

    main(parser.parse_args())
